import tempfile
from pathlib import Path

import pytest

from my_project_orchestrator.core.llm_cache import LLMCache


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
