"""Harvest OpenRouter's rotating free models.

OpenRouter publishes a public model list whose ``:free`` entries (zero prompt
and completion price) rotate over time. This fetches and day-caches that list
so the selection policy can offer free models as cheap-tier candidates.

Reliability is preserved upstream of this module: a free model only ever
appears on non-final attempts, is subject to the same proven-first-try gate as
any cheap model, and (when a paid failover is configured) degrades to the paid
provider on rate-limit/outage. So harvesting can only reduce cost, never lower
the quality floor enforced by the validation gates.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)

_MODELS_URL = "https://openrouter.ai/api/v1/models"
_DAY_SECONDS = 24 * 3600
# Bound the body materialized from the (semi-trusted) OpenRouter endpoint. Real
# model lists are well under this; the cap only rejects a compromised/misbehaving
# endpoint slow-dripping an oversized body. Callers already treat a fetch that
# raises as a transient failure and fall back to the cached list.
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024

# Per-process cache of fetched list endpoints, keyed by URL. Free-model
# harvesting, the model catalog, and each failover client's catalog all hit the
# same /models endpoint; this collapses those into one network round-trip per
# URL per run. Successful results only (failures re-raise without caching).
_FETCH_CACHE: Dict[str, list] = {}


def _http_fetch(url: str = _MODELS_URL) -> list:
    """Fetch and return the ``data`` array from an OpenRouter list endpoint.

    Shared by free-model harvesting, the model catalog, and embedding-model
    discovery — the network boundary lives in exactly one place.
    """
    if url in _FETCH_CACHE:
        return _FETCH_CACHE[url]
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "misterdev"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read(_MAX_RESPONSE_BYTES + 1)
    if len(raw) > _MAX_RESPONSE_BYTES:
        raise ValueError(f"model-list response exceeded {_MAX_RESPONSE_BYTES} bytes")
    payload = json.loads(raw.decode("utf-8"))
    data = payload.get("data", []) if isinstance(payload, dict) else []
    _FETCH_CACHE[url] = data
    return data


def _is_free(model: dict) -> bool:
    pricing = model.get("pricing") or {}

    def _zero(value) -> bool:
        try:
            return float(value) == 0.0
        except (TypeError, ValueError):
            return False

    # A model is free only when both prompt and completion are explicitly zero.
    return (
        "prompt" in pricing
        and "completion" in pricing
        and _zero(pricing.get("prompt"))
        and _zero(pricing.get("completion"))
    )


class FreeModelCache:
    """Day-cached list of free OpenRouter model ids, backed by a JSON file."""

    def __init__(self, path: Path, fetcher: Optional[Callable[[], list]] = None):
        self.path = Path(path)
        self._fetcher = fetcher or _http_fetch

    def _load(self) -> Optional[dict]:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (json.JSONDecodeError, OSError):
            return None

    def _save(self, data: dict) -> None:
        from misterdev.utils.file_utils import atomic_write_json

        atomic_write_json(self.path, data, indent=2, sort_keys=True)

    def get(self, now: float, max_age_seconds: int = _DAY_SECONDS) -> List[str]:
        """Return current free model ids, refetching when the cache is stale.

        On a fetch failure, falls back to the last cached list (possibly empty)
        so a transient network problem never blocks a build.
        """
        cached = self._load()
        if cached and (now - cached.get("fetched_at", 0)) < max_age_seconds:
            return list(cached.get("models", []))
        try:
            raw = self._fetcher()
        except Exception as e:
            logger.warning(f"Free-model fetch failed, using cached list: {e}")
            return list(cached.get("models", [])) if cached else []
        free = sorted(
            m["id"] for m in raw if isinstance(m, dict) and m.get("id") and _is_free(m)
        )
        self._save({"fetched_at": now, "models": free})
        logger.info(f"Harvested {len(free)} free OpenRouter models")
        return free
