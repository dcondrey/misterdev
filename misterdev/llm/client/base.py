import random
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

from misterdev.config import get_setting
from misterdev.logging_setup import setup_logger

from .errors import BudgetExceededError, LLMCallError
from .rate_coordinator import record_rate_limit, wait_if_needed
from .response import LLMResponse, LLMUsage

logger = setup_logger(__name__)

# Hard ceiling on the characters sent in a single prompt, a last-resort guard so
# no caller can ever request a prompt larger than any model's context window and
# hard-fail the whole task (a context-assembly bug once produced a ~37MB / 9M-token
# prompt). ~2.8M chars ≈ ~800k tokens, under a 1M-token window with headroom.
# Generous on purpose: normal prompts are budgeted far below this, so it only ever
# fires on a pathological blowup. When it fires, the MIDDLE is elided (the head
# carries instructions, the tail carries the actual ask), which degrades far more
# gracefully than a 400.
_MAX_PROMPT_CHARS = 2_400_000


def _bound_prompt(prompt: str) -> str:
    """Middle-truncate a prompt that exceeds the hard char ceiling.

    Keeps the head and tail (instructions + the ask) and elides the middle with a
    marker. A no-op for every normal prompt; only a runaway context-assembly bug
    can trip it, and truncating beats a context-length API error that fails the
    task outright.
    """
    if len(prompt) <= _MAX_PROMPT_CHARS:
        return prompt
    keep = _MAX_PROMPT_CHARS // 2
    omitted = len(prompt) - 2 * keep
    logger.warning(
        f"Prompt of {len(prompt)} chars exceeds the {_MAX_PROMPT_CHARS} ceiling; "
        f"eliding {omitted} chars from the middle (context-assembly overflow)."
    )
    return (
        prompt[:keep]
        + f"\n\n... [{omitted} characters elided to fit the model context] ...\n\n"
        + prompt[-keep:]
    )


class BaseLLMClient(ABC):
    """Abstract LLM client with token tracking and budget enforcement."""

    # Fraction of budget-remaining a single task may spend when its per-task
    # cap is set to "auto".
    AUTO_TASK_CAP_FRACTION = 0.5
    # Floor for the "auto" cap: below this, a task defers before it can write a
    # working implementation. On a small budget (or after pre-execution
    # overhead), fraction*remaining falls under the ~$0.2-0.4 a real task costs,
    # so a substantial task (e.g. a from-scratch Bowling game) is abandoned with
    # its stub untouched. Floor at a minimum-viable budget, bounded by what is
    # actually left, so the fraction guard still binds on large budgets.
    MIN_TASK_BUDGET = 0.40

    def __init__(self, config: dict):
        self.config = config
        self.cumulative_usage = LLMUsage()
        # Guards the usage accumulators: one client instance is hit by several
        # worker threads at once (parallel analyzers, executor waves), so the
        # read-modify-write of cost/token counters must be serialized or
        # concurrent calls lose updates and under-count spend.
        self._usage_lock = threading.Lock()
        # Per-task ROUTING/ATTRIBUTION state (active model, current task, reasoning
        # effort, edit-tool mode) is thread-local: parallel executor workers share
        # ONE client instance (see agent._WorktreeProjectView), so a plain instance
        # attribute would let one worker's with_model()/track_task() clobber
        # another's mid-call — issuing a request to the wrong model or attributing
        # its cost to the wrong task. The shared accounting below stays shared.
        self._tls = threading.local()
        self._default_model: Optional[str] = None
        self.cost_by_task: Dict[str, float] = {}
        self._budget = get_setting(config, "build", "budget")
        # Optional per-task cost ceiling. None disables it; a number is an
        # absolute cap; "auto" makes it a fraction of the budget remaining when
        # the task starts (snapshotted in track_task), so it shrinks as the run
        # spends and no single task can drain the global budget.
        self._max_cost_per_task = get_setting(
            config, "orchestrator", "max_cost_per_task"
        )
        self._task_caps: Dict[str, Optional[float]] = {}

    @property
    def model(self) -> Optional[str]:
        """Active model: a thread-local ``with_model`` override, else the
        configured default (set once at construction, shared across threads).

        Tolerates a partially-constructed instance (``_tls``/``_default_model``
        absent) so unit tests that build a client via ``__new__`` to exercise a
        pure method still read ``model``."""
        override = getattr(getattr(self, "_tls", None), "model", None)
        return override or getattr(self, "_default_model", None)

    @model.setter
    def model(self, value: Optional[str]) -> None:
        # Construction sets the shared default; per-call overrides go through
        # with_model() (thread-local), never this setter.
        self._default_model = value

    @property
    def _current_task(self) -> Optional[str]:
        return getattr(self._tls, "current_task", None)

    @_current_task.setter
    def _current_task(self, value: Optional[str]) -> None:
        self._tls.current_task = value

    @property
    def _reasoning_effort(self) -> Optional[str]:
        return getattr(self._tls, "reasoning_effort", None)

    @_reasoning_effort.setter
    def _reasoning_effort(self, value: Optional[str]) -> None:
        self._tls.reasoning_effort = value

    @property
    def _edit_tool_mode(self) -> bool:
        return getattr(self._tls, "edit_tool_mode", False)

    @_edit_tool_mode.setter
    def _edit_tool_mode(self, value: bool) -> None:
        self._tls.edit_tool_mode = value

    def _prepare_prompt(self, prompt: str) -> str:
        """Enforce the budget and bound the prompt size in one place.

        The single invariant that every outbound prompt is BOTH budget-checked and
        size-capped: a new entry point calls this instead of repeating the pair and
        risking one being satisfied while the other is forgotten.
        """
        self._enforce_budget()
        return _bound_prompt(prompt)

    def generate(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        """Generate a response with retry and budget enforcement.

        This is the primary public interface. Subclasses implement _call().
        """
        prompt = self._prepare_prompt(prompt)

        # Free models (`:free`) are frequently rate-limited upstream and slow to
        # even return the 429. Since routed calls fall back to the reliable paid
        # model on failure, a free model should FAIL FAST (one shot) rather than
        # burn ~minutes on slow retries before that fallback kicks in. Paid models
        # keep full retry resilience for genuine transient errors.
        is_free = ":free" in (getattr(self, "model", "") or "")
        max_retries = 1 if is_free else 3
        base_delay = 1.0
        model_key = getattr(self, "model", "") or ""

        if model_key:
            wait_if_needed(model_key)

        last_error = None
        for attempt in range(max_retries):
            try:
                response = self._call(prompt, system_prompt)
                self._track_usage(response.usage)
                return response
            except LLMCallError as e:
                last_error = e
                if not e.retryable or attempt == max_retries - 1:
                    raise
                delay = base_delay * (2**attempt) * random.uniform(0.8, 1.2)
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {delay:.1f}s: {e}"
                )
                if model_key:
                    record_rate_limit(model_key, delay)
                    wait_if_needed(model_key)
                else:
                    time.sleep(delay)

        raise last_error

    def generate_code(self, prompt: str, system_prompt: str = "") -> str:
        """Convenience wrapper returning just the text content.

        Maintains backward compatibility with existing callers.
        """
        return self.generate(prompt, system_prompt).content

    def generate_edits(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        """Generate file edits, preferring a structured tool call when supported.

        The base implementation just calls :meth:`generate` (markdown edits);
        clients that support function-calling override to force the apply_edits
        tool and return content rendered into the canonical fence format.
        """
        return self.generate(prompt, system_prompt)

    def chat_multimodal(
        self, prompt: str, image_b64: str, model: Optional[str] = None
    ) -> str:
        """Send a text+image message and return the model's text reply.

        First-class multimodal entry point used by the vision gate. The base
        raises so providers without a multimodal path degrade cleanly (the
        vision gate then falls back / SKIPs rather than crashing).
        """
        raise NotImplementedError("multimodal chat not supported by this client")

    def health_check(self) -> Tuple[bool, str]:
        """Verify the configured model actually resolves before a real run.

        A retired/misrouted model id (e.g. an OpenRouter 404) otherwise only
        surfaces on the first analysis call, after preflight has already
        spent setup time. This makes one minimal request and reports failure
        with the model id so the build can abort early with a clear message.
        """
        model = getattr(self, "model", "<unknown>")
        prior = self.cumulative_usage.estimated_cost
        try:
            self.generate("ping", "Reply with the single word OK.")
            return True, model
        except Exception as e:
            return False, f"model {model!r} unavailable: {e}"
        finally:
            self.cumulative_usage.estimated_cost = prior

    @contextmanager
    def with_model(self, model: str):
        """Temporarily override the active model for per-task routing.

        The override is thread-local (a shared client is used by parallel
        workers), so it must set/restore ``_tls.model`` directly rather than the
        ``model`` setter, which writes the shared default.
        """
        had_override = hasattr(self._tls, "model")
        previous = getattr(self._tls, "model", None)
        self._tls.model = model
        try:
            yield
        finally:
            if had_override:
                self._tls.model = previous
            else:
                try:
                    del self._tls.model
                except AttributeError:
                    pass

    @contextmanager
    def with_reasoning_effort(self, effort: Optional[str]):
        """Temporarily request a reasoning effort for the enclosed call(s).

        Honored only by clients/models that support a reasoning budget; ignored
        elsewhere. ``None`` is a no-op.
        """
        original = getattr(self, "_reasoning_effort", None)
        self._reasoning_effort = effort
        try:
            yield
        finally:
            self._reasoning_effort = original

    def _enforce_budget(self) -> None:
        """Raise BudgetExceededError if the global or per-task cap is spent.

        Shared by generate() and generate_stream() so both paths honor the
        same budget gates before issuing a call.
        """
        # Read the cumulative cost under the same lock that _track_usage writes
        # it, so a parallel worker mid-update can't be observed as a torn/stale
        # value that lets a call slip past the ceiling.
        with self._usage_lock:
            spent = self.cumulative_usage.estimated_cost
        if spent >= self._budget:
            raise BudgetExceededError(
                f"Budget of ${self._budget:.2f} exceeded (spent ${spent:.2f})"
            )
        if self.task_cost_exceeded(getattr(self, "_current_task", None)):
            task_id = getattr(self, "_current_task", None)
            cap = self.effective_task_cap(task_id)
            raise BudgetExceededError(
                f"Per-task budget of ${cap:.2f} exceeded "
                f"for task {task_id!r} "
                f"(spent ${self.task_cost(task_id):.2f})"
            )

    @abstractmethod
    def _call(self, prompt: str, system_prompt: str) -> LLMResponse:
        """Execute a single LLM API call. Subclasses implement this."""

    def generate_stream(
        self, prompt: str, system_prompt: str = "", abort_check=None
    ) -> LLMResponse:
        """Stream a response, aborting early if abort_check flags the output.

        Returns finish_reason="aborted" with the partial content when the check
        trips, so a caller can retry with a stricter prompt instead of waiting
        for a full bad response. Usage is captured from the provider stream when
        available (and estimated otherwise) so streaming honors the same budget
        accounting as generate().
        """
        prompt = self._prepare_prompt(prompt)

        content = ""
        usage: Optional[LLMUsage] = None
        finish_reason = "stop"
        stream = self._call_stream(prompt, system_prompt)
        try:
            while True:
                content += next(stream)
                if abort_check is not None and abort_check(content):
                    logger.warning("Aborting LLM stream: bad output pattern detected")
                    stream.close()
                    finish_reason = "aborted"
                    break
        except StopIteration as stop:
            usage = stop.value
        if usage is None:
            usage = self._estimate_usage(prompt, system_prompt, content)
        self._track_usage(usage)
        return LLMResponse(
            content=content,
            usage=usage,
            model=getattr(self, "model", ""),
            finish_reason=finish_reason,
        )

    def _call_stream(self, prompt: str, system_prompt: str):
        raise NotImplementedError("streaming not supported by this client")
        yield  # pragma: no cover - marks this a generator for callers

    def _estimate_usage(
        self, prompt: str, system_prompt: str, content: str
    ) -> LLMUsage:
        """Approximate usage when a stream yields no API usage data.

        Used on early abort (the stream is abandoned before its usage chunk)
        or with providers that omit usage. Tokens are estimated at ~4 chars
        each; cost defers to the provider's _estimate_cost.
        """
        prompt_tokens = (len(prompt) + len(system_prompt)) // 4
        completion_tokens = len(content) // 4
        return LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            estimated_cost=self._estimate_cost(prompt_tokens, completion_tokens),
        )

    def _estimate_cost(
        self, prompt_tokens: int, completion_tokens: int, *args
    ) -> float:
        """Dollar cost for a token count. Overridden per provider."""
        return 0.0

    @contextmanager
    def track_task(self, task_id: str):
        """Attribute LLM cost/calls made in this block to a task id."""
        previous = getattr(self, "_current_task", None)
        self._current_task = task_id
        # Snapshot the per-task cost cap on first entry so an "auto" cap is
        # fixed to the budget available when the task started, not recomputed
        # (and shrinking) on every call as the task spends.
        if task_id is not None and task_id not in self._task_caps:
            self._task_caps[task_id] = self._resolve_task_cap()
        try:
            yield
        finally:
            self._current_task = previous

    def _resolve_task_cap(self) -> Optional[float]:
        """Resolve the configured per-task cap to a dollar figure (or None)."""
        raw = self._max_cost_per_task
        if isinstance(raw, bool) or raw is None:
            return None
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str) and raw.strip().lower() == "auto":
            frac = self.budget_remaining * self.AUTO_TASK_CAP_FRACTION
            return max(frac, min(self.budget_remaining, self.MIN_TASK_BUDGET))
        return None

    def effective_task_cap(self, task_id: Optional[str]) -> Optional[float]:
        """The dollar cap for a task, or None when uncapped.

        Prefers the value snapshotted by ``track_task``; falls back to a live
        resolve so the cap is meaningful even when queried before the task
        block is entered.
        """
        if task_id is None:
            return None
        if task_id in self._task_caps:
            return self._task_caps[task_id]
        return self._resolve_task_cap()

    def _track_usage(self, usage: LLMUsage) -> None:
        bucket = getattr(self, "_current_task", None) or "overhead"
        with self._usage_lock:
            self.cumulative_usage.prompt_tokens += usage.prompt_tokens
            self.cumulative_usage.completion_tokens += usage.completion_tokens
            self.cumulative_usage.total_tokens += usage.total_tokens
            self.cumulative_usage.estimated_cost += usage.estimated_cost
            self.cumulative_usage.call_count += 1
            self.cumulative_usage.cache_creation_tokens += usage.cache_creation_tokens
            self.cumulative_usage.cache_read_tokens += usage.cache_read_tokens
            self.cost_by_task[bucket] = (
                self.cost_by_task.get(bucket, 0.0) + usage.estimated_cost
            )
            running = self.cumulative_usage.estimated_cost
        logger.debug(
            "llm call model=%s bucket=%s tokens=%d/%d cost=$%.4f cumulative=$%.4f",
            getattr(self, "model", ""),
            bucket,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.estimated_cost,
            running,
        )

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self._budget - self.cumulative_usage.estimated_cost)

    def task_cost(self, task_id: Optional[str]) -> float:
        """Accumulated cost attributed to a task id (0.0 if untracked)."""
        if task_id is None:
            return 0.0
        return getattr(self, "cost_by_task", {}).get(task_id, 0.0)

    def task_cost_exceeded(self, task_id: Optional[str]) -> bool:
        """True when task_id has crossed its snapshotted per-task cost cap."""
        if task_id is None:
            return False
        cap = self.effective_task_cap(task_id)
        if cap is None:
            return False
        return self.task_cost(task_id) >= cap
