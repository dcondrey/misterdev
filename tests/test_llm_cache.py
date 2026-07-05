import os
import tempfile
from pathlib import Path

import pytest

from misterdev.core.economics.llm_cache import LLMCache


@pytest.fixture
def cache():
    with tempfile.TemporaryDirectory() as d:
        yield LLMCache(Path(d) / "llm_cache")


def test_miss_returns_none(cache):
    assert cache.get("sys", "prompt") is None


def test_put_then_get_roundtrip(cache):
    cache.put("sys", "prompt", "the output", model="free/x", timestamp=1.0)
    assert cache.get("sys", "prompt") == "the output"


def test_key_depends_on_both_prompts(cache):
    cache.put("sys", "prompt", "out1")
    # Different system prompt -> different key -> miss.
    assert cache.get("other-sys", "prompt") is None
    # Different user prompt -> miss.
    assert cache.get("sys", "other-prompt") is None


def test_changed_input_invalidates(cache):
    cache.put("sys", "code v1", "edit-for-v1")
    # When the embedded file content changes, the prompt changes, so the old
    # entry is not returned for the new prompt.
    assert cache.get("sys", "code v2") is None
    assert cache.get("sys", "code v1") == "edit-for-v1"


def test_corrupt_entry_is_ignored(cache):
    cache.put("sys", "prompt", "good")
    key = cache._key("sys", "prompt")
    cache._path(key).write_text("{broken", encoding="utf-8")
    assert cache.get("sys", "prompt") is None


def test_overwrite_updates_entry(cache):
    cache.put("sys", "p", "first")
    cache.put("sys", "p", "second")
    assert cache.get("sys", "p") == "second"


def _count_entries(cache):
    return len(list(cache.dir.glob("*.json")))


def test_eviction_caps_entry_count():
    with tempfile.TemporaryDirectory() as d:
        cache = LLMCache(Path(d) / "c", max_entries=5)
        for i in range(20):
            # mtime granularity is coarse; stamp each file so eviction order is
            # well-defined regardless of clock resolution.
            cache.put("sys", f"prompt-{i}", f"out-{i}")
            p = cache._path(cache._key("sys", f"prompt-{i}"))
            os.utime(p, (i, i))
        assert _count_entries(cache) == 5


def test_eviction_drops_oldest_first():
    with tempfile.TemporaryDirectory() as d:
        cache = LLMCache(Path(d) / "c", max_entries=3)
        for i in range(6):
            cache.put("sys", f"p{i}", f"o{i}")
            os.utime(cache._path(cache._key("sys", f"p{i}")), (i, i))
        # Oldest (p0..p2) evicted; newest (p3..p5) retained.
        assert cache.get("sys", "p0") is None
        assert cache.get("sys", "p2") is None
        assert cache.get("sys", "p5") == "o5"


def test_zero_max_entries_disables_eviction():
    with tempfile.TemporaryDirectory() as d:
        cache = LLMCache(Path(d) / "c", max_entries=0)
        for i in range(10):
            cache.put("sys", f"p{i}", f"o{i}")
        assert _count_entries(cache) == 10


def test_eviction_ignores_tmp_files():
    with tempfile.TemporaryDirectory() as d:
        cache = LLMCache(Path(d) / "c", max_entries=100)
        cache.put("sys", "p", "o")
        # A stray .tmp file must not be counted or evicted as a cache entry.
        stray = cache.dir / "leftover.json.tmp"
        stray.write_text("{}", encoding="utf-8")
        cache.put("sys", "p2", "o2")
        assert stray.exists()
