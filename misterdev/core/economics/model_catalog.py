"""Per-model capability profiles from OpenRouter's model list.

Different models accept different parameters: a reasoning model rejects
``temperature``, an o-series model wants ``max_completion_tokens`` instead of
``max_tokens``, and only some models support a ``reasoning`` effort budget.
OpenRouter's ``/api/v1/models`` response carries a ``supported_parameters``
array, ``context_length``, and ``top_provider.max_completion_tokens`` per model
— this reads them so a request only ever sends parameters a model accepts.

Fetched once per process and cached in memory; failures degrade to an empty
catalog (callers then fall back to their existing default parameters).
"""

from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Optional

from misterdev.core.economics.free_models import _http_fetch
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)


@dataclass(frozen=True)
class ModelProfile:
    id: str
    supported_parameters: FrozenSet[str]
    max_completion_tokens: Optional[int] = None
    context_length: Optional[int] = None

    def supports(self, param: str) -> bool:
        return param in self.supported_parameters

    @property
    def supports_reasoning(self) -> bool:
        return (
            "reasoning" in self.supported_parameters
            or "reasoning_effort" in self.supported_parameters
        )


def _parse(raw: list) -> Dict[str, ModelProfile]:
    profiles: Dict[str, ModelProfile] = {}
    for m in raw:
        if not isinstance(m, dict):
            continue
        model_id = m.get("id")
        if not model_id:
            continue
        params = m.get("supported_parameters") or []
        top = m.get("top_provider")
        max_out = top.get("max_completion_tokens") if isinstance(top, dict) else None
        profiles[model_id] = ModelProfile(
            id=model_id,
            supported_parameters=frozenset(p for p in params if isinstance(p, str)),
            max_completion_tokens=max_out,
            context_length=m.get("context_length"),
        )
    return profiles


class ModelCatalog:
    """In-memory, lazily-fetched map of model id -> ModelProfile."""

    def __init__(self, fetcher: Optional[Callable[[], list]] = None):
        self._fetcher = fetcher or _http_fetch
        self._profiles: Optional[Dict[str, ModelProfile]] = None

    def _ensure(self) -> Dict[str, ModelProfile]:
        if self._profiles is None:
            try:
                self._profiles = _parse(self._fetcher())
            except Exception as e:
                logger.warning(f"Model catalog fetch failed; no profiles: {e}")
                self._profiles = {}
        return self._profiles

    def profile(self, model_id: str) -> Optional[ModelProfile]:
        """Profile for a model, or None when unknown (caller uses its defaults)."""
        return self._ensure().get(model_id)
