from misterdev.core.economics.model_catalog import (
    ModelCatalog,
    ModelProfile,
    _parse,
)


_SAMPLE = [
    {
        "id": "anthropic/claude-sonnet-4-6",
        "supported_parameters": ["temperature", "max_tokens", "tools"],
        "top_provider": {"max_completion_tokens": 8192},
        "context_length": 200000,
    },
    {
        "id": "openai/o3",
        "supported_parameters": ["max_completion_tokens", "reasoning_effort"],
        "top_provider": {"max_completion_tokens": 100000},
        "context_length": 128000,
    },
]


def test_profile_supports_and_reasoning():
    p = ModelProfile(
        id="x",
        supported_parameters=frozenset({"temperature", "reasoning"}),
    )
    assert p.supports("temperature")
    assert not p.supports("max_completion_tokens")
    assert p.supports_reasoning


def test_profile_reasoning_effort_alias():
    p = ModelProfile(id="x", supported_parameters=frozenset({"reasoning_effort"}))
    assert p.supports_reasoning


def test_profile_no_reasoning():
    p = ModelProfile(id="x", supported_parameters=frozenset({"temperature"}))
    assert not p.supports_reasoning


def test_parse_extracts_fields():
    profiles = _parse(_SAMPLE)
    sonnet = profiles["anthropic/claude-sonnet-4-6"]
    assert sonnet.max_completion_tokens == 8192
    assert sonnet.context_length == 200000
    assert sonnet.supports("tools")
    assert not sonnet.supports_reasoning
    assert profiles["openai/o3"].supports_reasoning


def test_parse_skips_non_dict_and_missing_id():
    profiles = _parse(["not a dict", {}, {"id": "ok", "supported_parameters": []}])
    assert list(profiles) == ["ok"]


def test_parse_handles_missing_top_provider():
    profiles = _parse([{"id": "m", "supported_parameters": ["max_tokens"]}])
    assert profiles["m"].max_completion_tokens is None
    assert profiles["m"].context_length is None


def test_parse_filters_non_string_params():
    profiles = _parse([{"id": "m", "supported_parameters": ["temperature", 5, None]}])
    assert profiles["m"].supported_parameters == frozenset({"temperature"})


def test_parse_handles_null_supported_parameters():
    profiles = _parse([{"id": "m", "supported_parameters": None}])
    assert profiles["m"].supported_parameters == frozenset()


def test_catalog_profile_lookup():
    cat = ModelCatalog(fetcher=lambda: _SAMPLE)
    assert cat.profile("openai/o3").supports_reasoning
    assert cat.profile("does/not-exist") is None


def test_catalog_fetches_once_and_caches():
    calls = {"n": 0}

    def fetcher():
        calls["n"] += 1
        return _SAMPLE

    cat = ModelCatalog(fetcher=fetcher)
    cat.profile("openai/o3")
    cat.profile("anthropic/claude-sonnet-4-6")
    cat.profile("missing")
    assert calls["n"] == 1


def test_catalog_fetch_failure_degrades_to_empty():
    def boom():
        raise RuntimeError("network down")

    cat = ModelCatalog(fetcher=boom)
    # Must not raise; unknown profile -> None so callers fall back to defaults.
    assert cat.profile("anything") is None


def test_catalog_fetch_failure_is_cached_not_retried():
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise RuntimeError("network down")

    cat = ModelCatalog(fetcher=boom)
    cat.profile("a")
    cat.profile("b")
    assert calls["n"] == 1
