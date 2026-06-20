import os
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Optional

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


def code_gen_abort_check(accumulated: str) -> bool:
    """Heuristic: True when a code-gen stream is clearly going wrong.

    Trips when a lot of text arrives with no code fence or file marker, or when
    the model opens with conversational filler instead of code.
    """
    if len(accumulated) > 2000 and "```" not in accumulated and "# File:" not in accumulated:
        return True
    head = accumulated[:200]
    return ("I'll help you" in head) or ("Sure, here" in head)


class BaseLLMClient(ABC):
    """Abstract LLM client with token tracking and budget enforcement."""

    def __init__(self, config: dict):
        self.config = config
        self.cumulative_usage = LLMUsage()
        self._budget = config.get("build", {}).get("budget", 100.0)

    def generate(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        """Generate a response with retry and budget enforcement.

        This is the primary public interface. Subclasses implement _call().
        """
        if self.cumulative_usage.estimated_cost >= self._budget:
            raise BudgetExceededError(
                f"Budget of ${self._budget:.2f} exceeded "
                f"(spent ${self.cumulative_usage.estimated_cost:.2f})"
            )

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
                delay = base_delay * (2 ** attempt)
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

    @contextmanager
    def with_model(self, model: str):
        """Temporarily override the active model (for per-task routing)."""
        original = getattr(self, "model", None)
        self.model = model
        try:
            yield
        finally:
            self.model = original

    @abstractmethod
    def _call(self, prompt: str, system_prompt: str) -> LLMResponse:
        """Execute a single LLM API call. Subclasses implement this."""

    def generate_stream(self, prompt: str, system_prompt: str = "", abort_check=None) -> LLMResponse:
        """Stream a response, aborting early if abort_check flags the output.

        Returns finish_reason="aborted" with the partial content when the check
        trips, so a caller can retry with a stricter prompt instead of waiting
        for a full bad response.
        """
        chunks: list[str] = []
        for chunk in self._call_stream(prompt, system_prompt):
            chunks.append(chunk)
            if abort_check is not None and abort_check("".join(chunks)):
                logger.warning("Aborting LLM stream: bad output pattern detected")
                return LLMResponse(content="".join(chunks),
                                   model=getattr(self, "model", ""), finish_reason="aborted")
        return LLMResponse(content="".join(chunks),
                           model=getattr(self, "model", ""), finish_reason="stop")

    def _call_stream(self, prompt: str, system_prompt: str):
        raise NotImplementedError("streaming not supported by this client")

    @contextmanager
    def track_task(self, task_id: str):
        """Attribute LLM cost/calls made in this block to a task id."""
        previous = getattr(self, "_current_task", None)
        self._current_task = task_id
        try:
            yield
        finally:
            self._current_task = previous

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
        self.cost_by_task[bucket] = self.cost_by_task.get(bucket, 0.0) + usage.estimated_cost

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self._budget - self.cumulative_usage.estimated_cost)


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

        env_var_name = llm_config.get("api_key_env_var", "OPENROUTER_API_KEY")
        self.api_key = os.environ.get(env_var_name)
        if not self.api_key:
            raise ValueError(f"API key environment variable '{env_var_name}' not set.")

        self.model = llm_config.get("model", "anthropic/claude-3.5-sonnet")
        self.temperature = llm_config.get("temperature", 0.1)

        from openai import OpenAI
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

    def _call(self, prompt: str, system_prompt: str) -> LLMResponse:
        logger.info(f"Calling OpenRouter model: {self.model}")
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )

            choice = response.choices[0]
            usage_data = response.usage

            usage = LLMUsage()
            if usage_data:
                usage.prompt_tokens = usage_data.prompt_tokens or 0
                usage.completion_tokens = usage_data.completion_tokens or 0
                usage.total_tokens = (usage.prompt_tokens + usage.completion_tokens)
                usage.estimated_cost = self._estimate_cost(
                    usage.prompt_tokens, usage.completion_tokens
                )

            return LLMResponse(
                content=choice.message.content or "",
                usage=usage,
                model=self.model,
                finish_reason=choice.finish_reason or "",
            )

        except Exception as e:
            error_str = str(e)
            retryable = any(s in error_str.lower() for s in [
                "rate limit", "timeout", "502", "503", "529",
                "overloaded", "connection",
            ])
            raise LLMCallError(f"OpenRouter API error: {e}", retryable=retryable) from e

    def _call_stream(self, prompt: str, system_prompt: str):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        stream = self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=self.temperature, stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        costs = self.COST_PER_1M.get(self.model, {"input": 3.0, "output": 15.0})
        return (
            (prompt_tokens / 1_000_000) * costs["input"]
            + (completion_tokens / 1_000_000) * costs["output"]
        )


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

        env_var_name = llm_config.get("api_key_env_var", "ANTHROPIC_API_KEY")
        self.api_key = os.environ.get(env_var_name)
        if not self.api_key:
            raise ValueError(f"API key environment variable '{env_var_name}' not set.")

        self.model = llm_config.get("model", "claude-sonnet-4-20250514")
        self.temperature = llm_config.get("temperature", 0.1)
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
                kwargs["system"] = [{
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }]

            response = self.client.messages.create(**kwargs)

            content = ""
            for block in response.content:
                if block.type == "text":
                    content += block.text

            cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
            cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
            usage = LLMUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                cache_creation_tokens=cache_creation,
                cache_read_tokens=cache_read,
                estimated_cost=self._estimate_cost(
                    response.usage.input_tokens, response.usage.output_tokens,
                    cache_creation, cache_read,
                ),
            )

            return LLMResponse(
                content=content,
                usage=usage,
                model=self.model,
                finish_reason=response.stop_reason or "",
            )

        except Exception as e:
            error_str = str(e)
            retryable = any(s in error_str.lower() for s in [
                "rate limit", "overloaded", "529", "timeout", "connection",
            ])
            raise LLMCallError(f"Anthropic API error: {e}", retryable=retryable) from e

    def _call_stream(self, prompt: str, system_prompt: str):
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        with self.client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                if text:
                    yield text

    def _estimate_cost(self, input_tokens: int, output_tokens: int,
                       cache_creation: int = 0, cache_read: int = 0) -> float:
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
        for fc in config.get("llm", {}).get("failover", []):
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


def _create_single_client(config: dict) -> BaseLLMClient:
    provider = config.get("llm", {}).get("provider", "openrouter")
    if provider == "openrouter":
        return OpenRouterLLMClient(config)
    elif provider == "anthropic":
        return AnthropicLLMClient(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def create_llm_client(config: dict) -> BaseLLMClient:
    """Factory: returns a failover-wrapped client when failover config exists."""
    if config.get("llm", {}).get("failover"):
        return FailoverLLMClient(config)
    return _create_single_client(config)
