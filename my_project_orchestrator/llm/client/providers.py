import json
import os
from typing import Optional

from my_project_orchestrator.config import get_section_setting
from my_project_orchestrator.logging_setup import setup_logger

from .base import BaseLLMClient
from .edits import APPLY_EDITS_TOOL, _edits_to_markdown
from .errors import _api_error
from .response import LLMResponse, LLMUsage

logger = setup_logger(__name__)

# Per-request network ceiling. The SDKs default to ~600s, so a hung/stalled
# connection can block a single attempt for ten minutes before retry/failover
# even kicks in. 300s bounds that while still allowing a large code generation
# to finish.
_REQUEST_TIMEOUT_SECONDS = 300.0


def _close_stream(stream) -> None:
    """Close an SDK stream's underlying HTTP connection (best effort), so an
    aborted generation doesn't leak the socket back into the connection pool."""
    close = getattr(stream, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


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

    return (
        OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        ),
        api_key,
    )


def _deny_unless_training_allowed(llm_config: dict) -> str:
    """OpenRouter data_collection policy: deny unless training is opted in."""
    if get_section_setting("llm", llm_config, "allow_training_models"):
        return "allow"
    return "deny"


# Per-1M-token price assumed for a model absent from a provider's COST_PER_1M
# table (a conservative mid-tier estimate so an unknown model is never costed
# as free). Shared by both providers' _estimate_cost.
_DEFAULT_COST_PER_1M = {"input": 3.0, "output": 15.0}


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

        from my_project_orchestrator.core.economics.model_catalog import ModelCatalog

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
        # Honor the budget gate and record spend just like generate()/_call() —
        # otherwise every vision call is invisible to the budget accounting and
        # cost_by_task, letting the cap be silently overrun.
        self._enforce_budget()
        try:
            resp = self.client.chat.completions.create(
                model=model or self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ],
                extra_body=self._extra_body(),
            )
            usage_data = getattr(resp, "usage", None)
            if usage_data:
                self._track_usage(self._usage_from(usage_data))
            return resp.choices[0].message.content or ""
        except Exception as e:
            raise _api_error("OpenRouter", e) from e

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

    @staticmethod
    def _build_messages(prompt: str, system_prompt: str) -> list:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _usage_from(self, usage_data) -> LLMUsage:
        """Build an LLMUsage from an OpenAI-compatible usage object (empty when
        the provider omitted usage)."""
        usage = LLMUsage()
        if usage_data:
            usage.prompt_tokens = usage_data.prompt_tokens or 0
            usage.completion_tokens = usage_data.completion_tokens or 0
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
            usage.estimated_cost = self._estimate_cost(
                usage.prompt_tokens, usage.completion_tokens
            )
        return usage

    def _call(self, prompt: str, system_prompt: str) -> LLMResponse:
        logger.info(f"Calling OpenRouter model: {self.model}")
        try:
            messages = self._build_messages(prompt, system_prompt)

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
            usage = self._usage_from(response.usage)

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
        messages = self._build_messages(prompt, system_prompt)
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
        try:
            for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
                usage_data = getattr(chunk, "usage", None)
                if usage_data:
                    usage = self._usage_from(usage_data)
        except Exception as e:
            # A drop DURING streaming was previously raised raw, bypassing the
            # LLMCallError classification that retry/failover keys on.
            raise _api_error("OpenRouter", e) from e
        finally:
            # Runs on normal completion, on a mid-stream error, AND on the
            # GeneratorExit raised when the caller aborts (code_gen_abort_check),
            # so the HTTP connection is always released.
            _close_stream(stream)
        return usage

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        # OpenRouter ":free" model ids are zero-cost. Without this, an unknown
        # (uncatalogued) free model falls through to the paid default below,
        # inflating budget accounting and making the cost-aware selector rank the
        # free models it exists to favor as the most expensive.
        if ":free" in self.model:
            return 0.0
        costs = self.COST_PER_1M.get(self.model, _DEFAULT_COST_PER_1M)
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

            self.client = anthropic.Anthropic(
                api_key=self.api_key, timeout=_REQUEST_TIMEOUT_SECONDS
            )
        except ImportError:
            raise ImportError(
                "anthropic package required for Anthropic provider. "
                "Install with: pip install anthropic"
            )

    def _usage_from(self, usage_data) -> LLMUsage:
        """Build an LLMUsage from an Anthropic usage object, including cache
        creation/read token accounting."""
        cache_creation = getattr(usage_data, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage_data, "cache_read_input_tokens", 0) or 0
        # Coalesce None to 0 like the OpenRouter sibling: a partial/aborted final
        # message can report null token counts, and the arithmetic below would
        # otherwise raise TypeError.
        input_tokens = usage_data.input_tokens or 0
        output_tokens = usage_data.output_tokens or 0
        return LLMUsage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
            estimated_cost=self._estimate_cost(
                input_tokens,
                output_tokens,
                cache_creation,
                cache_read,
            ),
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

            usage = self._usage_from(response.usage)

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

        try:
            with stream_cm as stream:
                for text in stream.text_stream:
                    if text:
                        yield text
                final = stream.get_final_message()
        except Exception as e:
            # A drop DURING streaming (or in get_final_message) was previously
            # raised raw, bypassing the LLMCallError classification that
            # retry/failover keys on — only the stream() call above was wrapped.
            # GeneratorExit (caller abort) is a BaseException, so it is not caught
            # here and still propagates while the with-block closes the stream.
            raise _api_error("Anthropic", e) from e

        return self._usage_from(final.usage)

    def _estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_creation: int = 0,
        cache_read: int = 0,
    ) -> float:
        costs = self.COST_PER_1M.get(self.model, _DEFAULT_COST_PER_1M)
        inp = costs["input"]
        # Anthropic pricing: cache reads ~10% of input, cache writes ~25% more.
        return (
            (input_tokens / 1_000_000) * inp
            + (cache_read / 1_000_000) * inp * 0.1
            + (cache_creation / 1_000_000) * inp * 1.25
            + (output_tokens / 1_000_000) * costs["output"]
        )
