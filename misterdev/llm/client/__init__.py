import time  # noqa: F401 — kept so tests can patch `client.time.sleep` on this package
from typing import Optional

from misterdev.config import get_setting
from misterdev.logging_setup import setup_logger

from .base import BaseLLMClient
from .edits import APPLY_EDITS_TOOL, _edits_to_markdown, code_gen_abort_check
from .embeddings import (
    LocalEmbeddingClient,
    OpenRouterEmbeddingClient,
    _create_local_embedding_client,
    _create_openrouter_embedding_client,
    create_embedding_client,
)
from .errors import (
    RETRYABLE_ERROR_MARKERS,
    RETRYABLE_EXCEPTION_NAMES,
    RETRYABLE_STATUS_CODES,
    BudgetExceededError,
    LLMCallError,
    _api_error,
    _error_status_code,
    _is_retryable_error,
)
from .providers import (
    CACHE_BREAKPOINT,
    AnthropicLLMClient,
    OpenRouterLLMClient,
    _deny_unless_training_allowed,
    _openrouter_sdk,
)
from .response import LLMResponse, LLMUsage

logger = setup_logger(__name__)


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


__all__ = [
    "APPLY_EDITS_TOOL",
    "CACHE_BREAKPOINT",
    "AnthropicLLMClient",
    "BaseLLMClient",
    "BudgetExceededError",
    "FailoverLLMClient",
    "LLMCallError",
    "LLMResponse",
    "LLMUsage",
    "LocalEmbeddingClient",
    "OpenRouterEmbeddingClient",
    "OpenRouterLLMClient",
    "RETRYABLE_ERROR_MARKERS",
    "RETRYABLE_EXCEPTION_NAMES",
    "RETRYABLE_STATUS_CODES",
    "_api_error",
    "_create_local_embedding_client",
    "_create_openrouter_embedding_client",
    "_create_single_client",
    "_deny_unless_training_allowed",
    "_edits_to_markdown",
    "_error_status_code",
    "_is_retryable_error",
    "_openrouter_sdk",
    "code_gen_abort_check",
    "create_embedding_client",
    "create_llm_client",
]
