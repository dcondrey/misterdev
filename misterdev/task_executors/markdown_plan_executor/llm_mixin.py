"""LLM invocation, routing, caching, and ledger recording."""

import contextlib
import time
from typing import Optional

from misterdev.core.models import Task
from misterdev.core.execution.project import Project
from misterdev.llm.client import code_gen_abort_check
from misterdev.llm.client.errors import _is_retryable_error
from misterdev.llm.client.rate_coordinator import is_on_cooldown, record_cooldown
from misterdev.config import get_setting

from .helpers import logger, _is_truncated


class LLMMixin:
    def _invoke_llm(self, project: Project, prompt: str, system_prompt: str):
        """Call the LLM, optionally streaming with early abort (config opt-in).

        Returns (text, aborted). Streaming is enabled via llm.streaming=true. A
        cache hit (llm.cache=true) returns the memoized output without a model
        call; it is still re-applied through the gates downstream.
        """
        cache = getattr(project, "llm_cache", None)
        if cache is not None:
            hit = cache.get(system_prompt, prompt)
            if hit is not None:
                logger.info("LLM cache hit; reusing prior output (no model call).")
                return hit, False
        if get_setting(project.config, "llm", "use_tools") and hasattr(
            project.llm_client, "generate_edits"
        ):
            # Structured edit extraction when the model supports tools; the
            # client renders the result into the canonical fence format and
            # falls back to plain generation for models without tool support.
            resp = project.llm_client.generate_edits(prompt, system_prompt)
            return resp.content, resp.finish_reason == "aborted"
        if get_setting(project.config, "llm", "streaming"):
            resp = project.llm_client.generate_stream(
                prompt, system_prompt, abort_check=code_gen_abort_check
            )
            return resp.content, resp.finish_reason == "aborted"
        return self._invoke_with_continuation(project, prompt, system_prompt), False

    def _invoke_with_continuation(
        self, project: Project, prompt: str, system_prompt: str
    ) -> str:
        """Plain-path generation that recovers a truncated response.

        Uses ``generate()`` (which exposes ``finish_reason``) instead of
        ``generate_code()``. When the response finishes normally this returns
        ``.content`` exactly as ``generate_code`` would, so the no-truncation
        path is byte-identical to before. When the model cuts the response off
        at its output-token limit, it issues up to ``orchestrator.max_continuations``
        bounded follow-up calls — each through the same client, so budget/usage
        tracking is never bypassed — asking the model to continue where it
        stopped, and concatenates the raw text before the edit parser sees it.
        The loop always halts: it stops on a normal finish_reason or at the cap.
        """
        resp = project.llm_client.generate(prompt, system_prompt)
        if not _is_truncated(resp.finish_reason):
            return resp.content

        max_continuations = get_setting(
            project.config, "orchestrator", "max_continuations"
        )
        parts = [resp.content]
        for _ in range(max(0, max_continuations)):
            logger.info(
                "Model response truncated (finish_reason="
                f"{resp.finish_reason!r}); requesting continuation."
            )
            cont_prompt = (
                f"{prompt}\n\n## Continuation\n"
                "Your previous output was cut off. Here is what you produced so "
                "far:\n\n" + "".join(parts) + "\n\nContinue the previous output "
                "exactly where it stopped. Do not repeat any earlier text and do "
                "not re-open code fences; emit only the remaining characters."
            )
            resp = project.llm_client.generate(cont_prompt, system_prompt)
            parts.append(resp.content)
            if not _is_truncated(resp.finish_reason):
                break
        return "".join(parts)

    def _cache_store(self, project, system_prompt, prompt, output, model) -> None:
        """Memoize a gate-passing output (no-op when caching is disabled)."""
        cache = getattr(project, "llm_cache", None)
        if cache is not None and output:
            try:
                cache.put(
                    system_prompt, prompt, output, model=model, timestamp=time.time()
                )
            except OSError as e:
                logger.warning(f"Failed to write LLM cache entry: {e}")

    def _resolve_model(
        self, project: Project, task: Task, strategy: str
    ) -> Optional[str]:
        """Pick a model for this task from llm.routing/llm.models config.

        Routes by task complexity first, then strategy. Returns None when no
        routing is configured so the client keeps its default model.
        """
        routing = get_setting(project.config, "llm", "routing")
        models = get_setting(project.config, "llm", "models")
        if not routing or not models:
            return None
        tier = routing.get(task.complexity) or routing.get(strategy)
        if not tier:
            return None
        return models.get(tier) or models.get("default")

    def _select_model(
        self,
        project: Project,
        task: Task,
        strategy: str,
        attempt: int,
        max_attempts: int,
    ) -> Optional[str]:
        """Resolve the model for this attempt.

        Prefers the ledger-driven dynamic policy when enabled; otherwise falls
        back to the static complexity/strategy routing. Returns None to keep the
        client's default model.
        """
        selector = getattr(project, "model_selector", None)
        if selector is not None and selector.enabled:
            chosen = selector.select(
                task.category, task.complexity, attempt, max_attempts
            )
            if chosen:
                return chosen
        return self._resolve_model(project, task, strategy)

    def _invoke_routed(
        self,
        project,
        task,
        prompt: str,
        system_prompt: str,
        routed_model: Optional[str],
        attempt: int,
        track_models: bool,
    ):
        """Invoke the LLM on the routed model, degrading to the default on failure.

        A routed cheap/free model can be unavailable (no provider permitted under
        the data_collection policy) or rate-limited. Rather than fail the task,
        record the routed model's failure (so the ledger deprioritizes it) and
        retry the same attempt on the client's default model. Returns
        (llm_response, aborted, pending_record).
        """

        def _cost() -> float:
            return project.llm_client.task_cost(task.id) if track_models else 0.0

        model_used = routed_model or getattr(project.llm_client, "model", "")
        cost_before = _cost()
        t_before = time.time()

        with self._reasoning_ctx(project, task):
            if not routed_model or is_on_cooldown(routed_model):
                if routed_model and is_on_cooldown(routed_model):
                    logger.info(
                        f"[{task.id}] skipping {routed_model} (on cooldown); "
                        "using default model."
                    )
                    model_used = getattr(project.llm_client, "model", "")
                    cost_before = _cost()
                resp, aborted = self._invoke_llm(project, prompt, system_prompt)
            else:
                logger.info(
                    f"[{task.id}] routing to {routed_model} ({task.complexity})"
                )
                try:
                    with project.llm_client.with_model(routed_model):
                        resp, aborted = self._invoke_llm(project, prompt, system_prompt)
                except Exception as routed_err:
                    logger.warning(
                        f"[{task.id}] routed model {routed_model} failed "
                        f"({routed_err}); falling back to the default model.",
                        exc_info=not _is_retryable_error(routed_err),
                    )
                    if not _is_retryable_error(routed_err):
                        record_cooldown(routed_model)
                        self._ledger_record(
                            project,
                            task,
                            self._pending(
                                routed_model, attempt, cost_before, t_before, False
                            ),
                            success=False,
                        )
                    model_used = getattr(project.llm_client, "model", "")
                    cost_before = _cost()
                    resp, aborted = self._invoke_llm(project, prompt, system_prompt)

        pending = self._pending(model_used, attempt, cost_before, t_before, aborted)
        return resp, aborted, pending

    @staticmethod
    def _pending(model, attempt, cost_before, t_before, aborted) -> dict:
        """Build a per-attempt ledger record (latency measured from t_before)."""
        return {
            "model": model,
            "attempt": attempt,
            "cost_before": cost_before,
            "latency": 0.0 if aborted else time.time() - t_before,
            "aborted": aborted,
        }

    def _reasoning_ctx(self, project, task):
        """Context manager requesting reasoning effort scaled to task complexity.

        Effort comes from the llm.reasoning_effort map (by complexity); the
        client only acts on it for models that support a reasoning budget. A
        no-op when no effort is configured or the client lacks the hook.
        """
        mapping = get_setting(project.config, "llm", "reasoning_effort") or {}
        effort = mapping.get(getattr(task, "complexity", None))
        hook = getattr(project.llm_client, "with_reasoning_effort", None)
        if effort and hook is not None:
            return hook(effort)
        return contextlib.nullcontext()

    def _ledger_record(
        self, project, task, pending: Optional[dict], *, success: bool
    ) -> None:
        """Record one attempt's outcome to the model ledger (no-op when off)."""
        selector = getattr(project, "model_selector", None)
        if pending is None or selector is None or not selector.enabled:
            return
        task_cost_fn = getattr(project.llm_client, "task_cost", None)
        if task_cost_fn is None:
            return
        cost = max(0.0, task_cost_fn(task.id) - pending["cost_before"])
        project.model_ledger.record(
            pending["model"],
            task.category,
            task.complexity,
            success=success,
            first_try=pending["attempt"] == 0,
            aborted=pending["aborted"],
            cost=cost,
            latency=pending["latency"],
            edit_failure=pending.get("had_edit_failure", False),
        )
