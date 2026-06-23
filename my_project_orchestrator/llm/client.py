import json
import os
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from my_project_orchestrator.config import get_section_setting, get_setting
from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)


@dataclass
class LLMUsage:
    """Token usage tracking for budget enforcement."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    call_count: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass
class LLMResponse:
    """Structured response from an LLM call."""

    content: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    model: str = ""
    finish_reason: str = ""


class BudgetExceededError(Exception):
    pass


class LLMCallError(Exception):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


# Substrings that mark a provider error as transient and worth retrying or
# failing over. Shared by every provider's _call and _call_stream so the
# retry/failover decision is identical on both paths.
RETRYABLE_ERROR_MARKERS = (
    "rate limit",
    "timeout",
    "502",
    "503",
    "529",
    "overloaded",
    "connection",
)


def _is_retryable_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in RETRYABLE_ERROR_MARKERS)


def _api_error(provider: str, error: Exception) -> "LLMCallError":
    """Wrap a provider exception as an LLMCallError with retryability classified."""
    return LLMCallError(
        f"{provider} API error: {error}", retryable=_is_retryable_error(error)
    )


# Structured tool for edit extraction. Forcing this (when a model supports
# `tools`) replaces brittle markdown-fence parsing: the model returns
# well-formed JSON we render back into the canonical fence format the executor
# already consumes, so nothing downstream changes.
APPLY_EDITS_TOOL = {
    "type": "function",
    "function": {
        "name": "apply_edits",
        "description": (
            "Write the complete final content of each file to create or modify "
            "to satisfy the task."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Project-relative file path.",
                            },
                            "content": {
                                "type": "string",
                                "description": "Full final content of the file.",
                            },
                        },
                        "required": ["path", "content"],
                    },
                }
            },
            "required": ["edits"],
        },
    },
}


def _edits_to_markdown(edits: list) -> str:
    """Render structured edits into the canonical ```lang:path fence format.

    The executor's parser keys on the ``path`` after the colon; the language
    token is re-derived from the path downstream, so a placeholder is fine.
    """
    blocks = []
    for edit in edits:
        if not isinstance(edit, dict):
            continue
        path = edit.get("path")
        content = edit.get("content", "")
        if path:
            blocks.append(f"```text:{path}\n{content}\n```")
    return "\n\n".join(blocks)


def code_gen_abort_check(accumulated: str) -> bool:
    """Heuristic: True when a code-gen stream is clearly going wrong.

    Trips when a lot of text arrives with no code fence or file marker, or when
    the model opens with conversational filler instead of code.
    """
    if (
        len(accumulated) > 2000
        and "```" not in accumulated
        and "# File:" not in accumulated
    ):
        return True
    head = accumulated[:200]
    return ("I'll help you" in head) or ("Sure, here" in head)


class BaseLLMClient(ABC):
    """Abstract LLM client with token tracking and budget enforcement."""

    # Fraction of budget-remaining a single task may spend when its per-task
    # cap is set to "auto".
    AUTO_TASK_CAP_FRACTION = 0.5

    def __init__(self, config: dict):
        self.config = config
        self.cumulative_usage = LLMUsage()
        self._budget = get_setting(config, "build", "budget")
        # Optional per-task cost ceiling. None disables it; a number is an
        # absolute cap; "auto" makes it a fraction of the budget remaining when
        # the task starts (snapshotted in track_task), so it shrinks as the run
        # spends and no single task can drain the global budget.
        self._max_cost_per_task = get_setting(
            config, "orchestrator", "max_cost_per_task"
        )
        self._task_caps: Dict[str, Optional[float]] = {}

    def generate(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        """Generate a response with retry and budget enforcement.

        This is the primary public interface. Subclasses implement _call().
        """
        self._enforce_budget()

        max_retries = 3
        base_delay = 1.0

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
                delay = base_delay * (2**attempt)
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {delay:.1f}s: {e}"
                )
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
        """Temporarily override the active model (for per-task routing)."""
        original = getattr(self, "model", None)
        self.model = model
        try:
            yield
        finally:
            self.model = original

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
        if self.cumulative_usage.estimated_cost >= self._budget:
            raise BudgetExceededError(
                f"Budget of ${self._budget:.2f} exceeded "
                f"(spent ${self.cumulative_usage.estimated_cost:.2f})"
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
        self._enforce_budget()

        chunks: list[str] = []
        usage: Optional[LLMUsage] = None
        finish_reason = "stop"
        stream = self._call_stream(prompt, system_prompt)
        try:
            while True:
                chunks.append(next(stream))
                if abort_check is not None and abort_check("".join(chunks)):
                    logger.warning("Aborting LLM stream: bad output pattern detected")
                    stream.close()
                    finish_reason = "aborted"
                    break
        except StopIteration as stop:
            usage = stop.value

        content = "".join(chunks)
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
            return self.budget_remaining * self.AUTO_TASK_CAP_FRACTION
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
        self.cumulative_usage.prompt_tokens += usage.prompt_tokens
        self.cumulative_usage.completion_tokens += usage.completion_tokens
        self.cumulative_usage.total_tokens += usage.total_tokens
        self.cumulative_usage.estimated_cost += usage.estimated_cost
        self.cumulative_usage.call_count += 1
        self.cumulative_usage.cache_creation_tokens += usage.cache_creation_tokens
        self.cumulative_usage.cache_read_tokens += usage.cache_read_tokens
        if not hasattr(self, "cost_by_task"):
            self.cost_by_task: Dict[str, float] = {}
        bucket = getattr(self, "_current_task", None) or "overhead"
        self.cost_by_task[bucket] = (
            self.cost_by_task.get(bucket, 0.0) + usage.estimated_cost
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


def _openrouter_sdk(llm_config: dict):
    """Build an OpenAI SDK client pointed at OpenRouter. Returns (client, api_key).

    Shared by the chat and embedding clients so the base URL, env-var lookup,
    and missing-key error live in one place.
    """
    env_var = get_section_setting("llm", llm_config, "api_key_env_var")
    api_key = os.environ.get(env_var)
    if not api_key:
        raise ValueError(f"API key environment variable '{env_var}' not set.")
    from openai import OpenAI

    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key), api_key


def _deny_unless_training_allowed(llm_config: dict) -> str:
    """OpenRouter data_collection policy: deny unless training is opted in."""
    if get_section_setting("llm", llm_config, "allow_training_models"):
        return "allow"
    return "deny"


class OpenRouterLLMClient(BaseLLMClient):
    """LLM client using OpenRouter API (OpenAI-compatible)."""

    # Approximate costs per 1M tokens for common models
    COST_PER_1M = {
        "anthropic/claude-opus-4-8": {"input": 15.0, "output": 75.0},
        "anthropic/claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
        "anthropic/claude-sonnet-4.6": {"input": 3.0, "output": 15.0},
        "anthropic/claude-haiku-4-5": {"input": 0.80, "output": 4.0},
        "anthropic/claude-sonnet-4": {"input": 3.0, "output": 15.0},
        "anthropic/claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
        "anthropic/claude-haiku-4": {"input": 0.80, "output": 4.0},
        "openai/gpt-4o": {"input": 2.5, "output": 10.0},
        "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "google/gemini-2.5-pro": {"input": 1.25, "output": 10.0},
    }

    def __init__(self, config: dict):
        super().__init__(config)
        llm_config = config.get("llm", {})
        self.client, self.api_key = _openrouter_sdk(llm_config)
        self.model = get_section_setting("llm", llm_config, "model")
        self.temperature = get_section_setting("llm", llm_config, "temperature")
        self.sampling = dict(get_section_setting("llm", llm_config, "sampling") or {})
        self.data_collection = _deny_unless_training_allowed(llm_config)

        from my_project_orchestrator.core.model_catalog import ModelCatalog

        self._catalog = ModelCatalog()

    def _sampling_kwargs(self) -> dict:
        """Sampling params for the active model, filtered by what it supports.

        temperature plus any configured ``llm.sampling`` knobs (top_p, top_k,
        min_p, repetition_penalty, ...) are emitted only when the model's
        OpenRouter ``supported_parameters`` include them, so an unsupported knob
        never 400s the request. Unknown model (catalog miss/offline) falls back
        to the prior behavior: temperature only.
        """
        candidates = {"temperature": self.temperature, **self.sampling}
        profile = self._catalog.profile(self.model)
        if profile is None:
            return {"temperature": self.temperature}
        return {k: v for k, v in candidates.items() if profile.supports(k)}

    def _supports_tools(self) -> bool:
        profile = self._catalog.profile(self.model)
        return profile is not None and profile.supports("tools")

    def _supports_reasoning(self) -> bool:
        profile = self._catalog.profile(self.model)
        return profile is not None and profile.supports_reasoning

    @staticmethod
    def _extract_tool_edits(choice) -> list:
        """Pull the apply_edits arguments out of a tool-call response.

        Tolerates malformed/empty arguments by returning [] (the executor then
        sees no edits and retries, exactly as with an empty markdown response).
        """
        edits = []
        for call in getattr(choice.message, "tool_calls", None) or []:
            fn = getattr(call, "function", None)
            if fn is None or fn.name != "apply_edits":
                continue
            try:
                args = json.loads(fn.arguments or "{}")
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(args, dict) and isinstance(args.get("edits"), list):
                edits.extend(args["edits"])
        return edits

    def generate_edits(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        """Force the apply_edits tool when the active model supports it.

        Reuses generate()'s budget/retry/usage machinery via an internal flag;
        falls back to plain markdown generation for models without tool support.
        """
        if not self._supports_tools():
            return self.generate(prompt, system_prompt)
        self._edit_tool_mode = True
        try:
            return self.generate(prompt, system_prompt)
        finally:
            self._edit_tool_mode = False

    def chat_multimodal(
        self, prompt: str, image_b64: str, model: Optional[str] = None
    ) -> str:
        """Send a text part plus a base64 PNG image and return the reply text.

        Builds an OpenAI-compatible multimodal message and selects ``model`` (a
        vision model id) when given, else the client's configured model. Provider
        routing prefs are reused so a multimodal call honors the same
        data_collection policy as every other request.
        """
        resp = self.client.chat.completions.create(
            model=model or self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            extra_body=self._extra_body(),
        )
        return resp.choices[0].message.content or ""

    def _extra_body(self) -> dict:
        """OpenRouter provider routing prefs.

        ``data_collection="deny"`` confines routing to providers that do not
        store or train on inputs — enforced server-side for every call, which is
        what makes harvesting free models safe. A free model with no compliant
        provider simply errors and the executor escalates to the next tier.

        Also carries a reasoning-effort budget when one is requested for the
        call and the active model supports it.
        """
        body = {"provider": {"data_collection": self.data_collection}}
        effort = getattr(self, "_reasoning_effort", None)
        if effort and self._supports_reasoning():
            body["reasoning"] = {"effort": effort}
        return body

    def _call(self, prompt: str, system_prompt: str) -> LLMResponse:
        logger.info(f"Calling OpenRouter model: {self.model}")
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            tool_kwargs = {}
            if getattr(self, "_edit_tool_mode", False):
                tool_kwargs = {
                    "tools": [APPLY_EDITS_TOOL],
                    "tool_choice": {
                        "type": "function",
                        "function": {"name": "apply_edits"},
                    },
                }

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_body=self._extra_body(),
                **self._sampling_kwargs(),
                **tool_kwargs,
            )

            choice = response.choices[0]
            usage_data = response.usage

            usage = LLMUsage()
            if usage_data:
                usage.prompt_tokens = usage_data.prompt_tokens or 0
                usage.completion_tokens = usage_data.completion_tokens or 0
                usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
                usage.estimated_cost = self._estimate_cost(
                    usage.prompt_tokens, usage.completion_tokens
                )

            content = choice.message.content or ""
            if tool_kwargs:
                # Render the structured tool call back into the canonical fence
                # format the executor's parser consumes.
                content = _edits_to_markdown(self._extract_tool_edits(choice))

            return LLMResponse(
                content=content,
                usage=usage,
                model=self.model,
                finish_reason=choice.finish_reason or "",
            )

        except Exception as e:
            raise _api_error("OpenRouter", e) from e

    def _call_stream(self, prompt: str, system_prompt: str):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                extra_body=self._extra_body(),
                **self._sampling_kwargs(),
            )
        except Exception as e:
            raise _api_error("OpenRouter", e) from e

        usage = LLMUsage()
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
            usage_data = getattr(chunk, "usage", None)
            if usage_data:
                usage.prompt_tokens = usage_data.prompt_tokens or 0
                usage.completion_tokens = usage_data.completion_tokens or 0
                usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
                usage.estimated_cost = self._estimate_cost(
                    usage.prompt_tokens, usage.completion_tokens
                )
        return usage

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        costs = self.COST_PER_1M.get(self.model, {"input": 3.0, "output": 15.0})
        return (prompt_tokens / 1_000_000) * costs["input"] + (
            completion_tokens / 1_000_000
        ) * costs["output"]


class AnthropicLLMClient(BaseLLMClient):
    """LLM client using the Anthropic API directly."""

    COST_PER_1M = {
        "claude-opus-4-8": {"input": 15.0, "output": 75.0},
        "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
        "claude-haiku-4-5": {"input": 0.80, "output": 4.0},
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "claude-haiku-4-20250414": {"input": 0.80, "output": 4.0},
        "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
    }

    def __init__(self, config: dict):
        super().__init__(config)
        llm_config = config.get("llm", {})

        env_var_name = get_section_setting("llm", llm_config, "api_key_env_var")
        self.api_key = os.environ.get(env_var_name)
        if not self.api_key:
            raise ValueError(f"API key environment variable '{env_var_name}' not set.")

        self.model = get_section_setting("llm", llm_config, "model")
        self.temperature = get_section_setting("llm", llm_config, "temperature")
        self.max_tokens = llm_config.get("max_tokens", 8192)

        try:
            import anthropic

            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "anthropic package required for Anthropic provider. "
                "Install with: pip install anthropic"
            )

    def _call(self, prompt: str, system_prompt: str) -> LLMResponse:
        logger.info(f"Calling Anthropic model: {self.model}")
        try:
            kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            }
            if system_prompt:
                # Mark the system prompt cacheable: tasks in a wave share it,
                # so subsequent calls read it from cache at ~10% of input cost.
                kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]

            response = self.client.messages.create(**kwargs)

            content = ""
            for block in response.content:
                if block.type == "text":
                    content += block.text

            cache_creation = (
                getattr(response.usage, "cache_creation_input_tokens", 0) or 0
            )
            cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
            usage = LLMUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                cache_creation_tokens=cache_creation,
                cache_read_tokens=cache_read,
                estimated_cost=self._estimate_cost(
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                    cache_creation,
                    cache_read,
                ),
            )

            return LLMResponse(
                content=content,
                usage=usage,
                model=self.model,
                finish_reason=response.stop_reason or "",
            )

        except Exception as e:
            raise _api_error("Anthropic", e) from e

    def _call_stream(self, prompt: str, system_prompt: str):
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        try:
            stream_cm = self.client.messages.stream(**kwargs)
        except Exception as e:
            raise _api_error("Anthropic", e) from e

        with stream_cm as stream:
            for text in stream.text_stream:
                if text:
                    yield text
            final = stream.get_final_message()

        usage_data = final.usage
        cache_creation = getattr(usage_data, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage_data, "cache_read_input_tokens", 0) or 0
        return LLMUsage(
            prompt_tokens=usage_data.input_tokens,
            completion_tokens=usage_data.output_tokens,
            total_tokens=usage_data.input_tokens + usage_data.output_tokens,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
            estimated_cost=self._estimate_cost(
                usage_data.input_tokens,
                usage_data.output_tokens,
                cache_creation,
                cache_read,
            ),
        )

    def _estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_creation: int = 0,
        cache_read: int = 0,
    ) -> float:
        costs = self.COST_PER_1M.get(self.model, {"input": 3.0, "output": 15.0})
        inp = costs["input"]
        # Anthropic pricing: cache reads ~10% of input, cache writes ~25% more.
        return (
            (input_tokens / 1_000_000) * inp
            + (cache_read / 1_000_000) * inp * 0.1
            + (cache_creation / 1_000_000) * inp * 1.25
            + (output_tokens / 1_000_000) * costs["output"]
        )


class FailoverLLMClient(BaseLLMClient):
    """Wraps a primary client plus ordered fallbacks for provider outages.

    On a retryable error from one provider it advances to the next; a
    non-retryable error (bad request) propagates immediately. Usage from
    whichever provider served the request is tracked on this wrapper.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.primary = _create_single_client(config)
        self.failover_clients: list[BaseLLMClient] = []
        for fc in get_setting(config, "llm", "failover"):
            try:
                merged = {**config, "llm": {**config["llm"], **fc, "failover": []}}
                self.failover_clients.append(_create_single_client(merged))
            except (ValueError, ImportError) as e:
                logger.warning(f"Failover provider unavailable, skipping: {e}")

    @property
    def model(self) -> str:
        return self.primary.model

    def with_model(self, model: str):
        # Route the primary path; fallbacks keep their configured models.
        return self.primary.with_model(model)

    def _call(self, prompt: str, system_prompt: str) -> LLMResponse:
        clients = [self.primary] + self.failover_clients
        last_error: Optional[LLMCallError] = None
        for i, client in enumerate(clients):
            try:
                response = client._call(prompt, system_prompt)
                if i > 0:
                    logger.info(
                        f"Failover successful via {client.__class__.__name__} ({client.model})."
                    )
                return response
            except LLMCallError as e:
                last_error = e
                if not e.retryable:
                    raise
                logger.warning(
                    f"Provider {client.__class__.__name__} failed ({e}); trying next provider..."
                )
        raise last_error or LLMCallError("All LLM providers failed", retryable=False)

    def _call_stream(self, prompt: str, system_prompt: str):
        clients = [self.primary] + self.failover_clients
        last_error: Optional[LLMCallError] = None
        for i, client in enumerate(clients):
            try:
                usage = yield from client._call_stream(prompt, system_prompt)
                if i > 0:
                    logger.info(
                        f"Failover stream via {client.__class__.__name__} ({client.model})."
                    )
                return usage
            except LLMCallError as e:
                last_error = e
                if not e.retryable:
                    raise
                logger.warning(
                    f"Provider {client.__class__.__name__} stream failed ({e}); "
                    "trying next provider..."
                )
        raise last_error or LLMCallError("All LLM providers failed", retryable=False)


class OpenRouterEmbeddingClient:
    """Embedding client over OpenRouter's OpenAI-compatible embeddings endpoint."""

    def __init__(self, config: dict, model: str):
        llm_config = config.get("llm", {})
        self.client, self.api_key = _openrouter_sdk(llm_config)
        self.model = model
        self.dimensions = get_section_setting("llm", llm_config, "embedding_dimensions")
        self.data_collection = _deny_unless_training_allowed(llm_config)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return one vector per input text, in input order."""
        kwargs = {
            "model": self.model,
            "input": texts,
            "extra_body": {"provider": {"data_collection": self.data_collection}},
        }
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        response = self.client.embeddings.create(**kwargs)
        ordered = sorted(response.data, key=lambda d: d.index)
        return [list(d.embedding) for d in ordered]


class LocalEmbeddingClient:
    """Embedding client backed by a local fastembed model (offline, no API key).

    The model is downloaded once and cached by fastembed, then runs on CPU via
    ONNX. Loading is lazy (first ``embed`` call) so merely constructing the
    client is cheap and failure (e.g. fastembed not installed) surfaces where
    the factory can fall back gracefully.
    """

    def __init__(self, config: dict):
        llm_config = config.get("llm", {})
        self.model = (
            get_section_setting("llm", llm_config, "local_embedding_model")
            or "BAAI/bge-small-en-v1.5"
        )
        self._embedder = None

    def _ensure(self):
        if self._embedder is None:
            from fastembed import TextEmbedding

            self._embedder = TextEmbedding(model_name=self.model)
        return self._embedder

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return one vector per input text, in input order."""
        embedder = self._ensure()
        return [list(vec) for vec in embedder.embed(list(texts))]


def _create_single_client(config: dict) -> BaseLLMClient:
    provider = get_setting(config, "llm", "provider")
    if provider == "openrouter":
        return OpenRouterLLMClient(config)
    elif provider == "anthropic":
        return AnthropicLLMClient(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def create_llm_client(config: dict) -> BaseLLMClient:
    """Factory: returns a failover-wrapped client when failover config exists."""
    if get_setting(config, "llm", "failover"):
        return FailoverLLMClient(config)
    return _create_single_client(config)


def _create_local_embedding_client(config: dict):
    try:
        return LocalEmbeddingClient(config)
    except Exception as e:
        logger.warning(f"Local embedding client unavailable: {e}")
        return None


def _create_openrouter_embedding_client(config: dict):
    if get_setting(config, "llm", "provider") != "openrouter":
        return None
    from my_project_orchestrator.core.embeddings import pick_embedding_model

    model = pick_embedding_model(
        get_setting(config, "llm", "embedding_model"),
        prefer=get_setting(config, "llm", "embedding_prefer"),
    )
    if not model:
        return None
    try:
        return OpenRouterEmbeddingClient(config, model)
    except Exception as e:
        logger.warning(f"Embedding client unavailable: {e}")
        return None


def create_embedding_client(config: dict):
    """Build an embedding client, or None when embeddings are unavailable.

    Backend (llm.embedding_backend): "local" forces fastembed; "openrouter"
    forces the API; "none" disables dense ranking; "auto" uses OpenRouter for an
    OpenRouter provider and otherwise falls back to a local fastembed model so
    semantic retrieval works offline without an API key. Any setup failure
    returns None, so retrieval degrades to lexical-only rather than breaking.
    """
    backend = get_setting(config, "llm", "embedding_backend")
    if backend == "none":
        return None
    if backend == "local":
        return _create_local_embedding_client(config)
    if backend == "openrouter":
        return _create_openrouter_embedding_client(config)
    # auto
    if get_setting(config, "llm", "provider") == "openrouter":
        return _create_openrouter_embedding_client(config)
    return _create_local_embedding_client(config)
