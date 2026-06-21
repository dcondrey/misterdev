import json
import tempfile
from pathlib import Path

import pytest

from my_project_orchestrator.core.free_models import FreeModelCache, _is_free
from my_project_orchestrator.core.model_ledger import ModelLedger
from my_project_orchestrator.core.model_selector import ModelSelector


@pytest.fixture
def cache_path():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d) / "free_models.json"


_RAW = [
    {"id": "vendor/free-a:free", "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "vendor/free-b:free", "pricing": {"prompt": "0.0", "completion": "0"}},
    {"id": "vendor/paid", "pricing": {"prompt": "0.000003", "completion": "0.000015"}},
    {"id": "vendor/no-pricing"},
]


def test_is_free_detection():
    assert _is_free(_RAW[0])
    assert _is_free(_RAW[1])
    assert not _is_free(_RAW[2])
    assert not _is_free(_RAW[3])


def test_get_filters_and_caches(cache_path):
    calls = []

    def fetcher():
        calls.append(1)
        return _RAW

    cache = FreeModelCache(cache_path, fetcher=fetcher)
    free = cache.get(now=1000.0)
    assert free == ["vendor/free-a:free", "vendor/free-b:free"]
    assert cache_path.exists()
    # Within the cache window, no refetch.
    cache.get(now=1000.0 + 3600)
    assert len(calls) == 1


def test_get_refetches_when_stale(cache_path):
    calls = []

    def fetcher():
        calls.append(1)
        return _RAW

    cache = FreeModelCache(cache_path, fetcher=fetcher)
    cache.get(now=0.0)
    cache.get(now=25 * 3600)  # past the 24h window
    assert len(calls) == 2


def test_fetch_failure_falls_back_to_cache(cache_path):
    cache_path.write_text(
        json.dumps({"fetched_at": 0, "models": ["vendor/old:free"]}), encoding="utf-8"
    )

    def boom():
        raise RuntimeError("network down")

    cache = FreeModelCache(cache_path, fetcher=boom)
    # Stale + fetch fails -> last known list, not a crash.
    assert cache.get(now=99 * 3600) == ["vendor/old:free"]


def test_fetch_failure_no_cache_returns_empty(cache_path):
    def boom():
        raise RuntimeError("network down")

    assert FreeModelCache(cache_path, fetcher=boom).get(now=0.0) == []


def test_selector_merges_free_into_cheapest_tier():
    with tempfile.TemporaryDirectory() as d:
        ledger = ModelLedger(Path(d) / "stats.json")
        config = {
            "llm": {
                "dynamic_selection": True,
                "escalation": ["cheap", "strong"],
                "models": {"cheap": "paid/cheap", "strong": "paid/strong"},
                "selection_posture": "aggressive",
            }
        }
        sel = ModelSelector(config, ledger, free_models=["vendor/free:free"])
        # Free model is now a candidate in the cheap tier and, being unseen,
        # wins the aggressive first attempt via the +inf UCB score.
        assert "vendor/free:free" in sel._tier_models("cheap")
        assert sel.select("feature", "small", 0, 3) == "vendor/free:free"
