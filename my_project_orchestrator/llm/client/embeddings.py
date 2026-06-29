from typing import List

from my_project_orchestrator.config import get_section_setting, get_setting
from my_project_orchestrator.logging_setup import setup_logger

from .providers import _deny_unless_training_allowed, _openrouter_sdk

logger = setup_logger(__name__)


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


def _create_local_embedding_client(config: dict):
    try:
        return LocalEmbeddingClient(config)
    except Exception as e:
        logger.warning(f"Local embedding client unavailable: {e}")
        return None


def _create_openrouter_embedding_client(config: dict):
    if get_setting(config, "llm", "provider") != "openrouter":
        return None
    from my_project_orchestrator.core.economics.embeddings import pick_embedding_model

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
