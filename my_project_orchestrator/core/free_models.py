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
from typing import Callable, List, Optional

from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)

_MODELS_URL = "https://openrouter.ai/api/v1/models"
_DAY_SECONDS = 24 * 3600


def _http_fetch() -> list:
    """Fetch the raw OpenRouter model list (network boundary)."""
    import urllib.request

    req = urllib.request.Request(
        _MODELS_URL, headers={"User-Agent": "project-orchestrator"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload.get("data", []) if isinstance(payload, dict) else []


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
        from my_project_orchestrator.utils.file_utils import ensure_artifact_dir

        ensure_artifact_dir(self.path.parent)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

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
