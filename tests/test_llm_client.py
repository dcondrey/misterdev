import json
import types

import pytest

from my_project_orchestrator.llm.client import (
    BaseLLMClient,
    FailoverLLMClient,
    LLMResponse,
    LLMUsage,
    LLMCallError,
    BudgetExceededError,
    code_gen_abort_check,
    create_llm_client,
    create_embedding_client,
    LocalEmbeddingClient,
)


def test_embedding_backend_routing():
    # local: always the fastembed client (construction is lazy, no download)
    c = create_embedding_client({"llm": {"provider": "anthropic", "embedding_backend": "local"}})
    assert isinstance(c, LocalEmbeddingClient)
    # auto + non-openrouter provider -> falls back to local
    c = create_embedding_client({"llm": {"provider": "anthropic", "embedding_backend": "auto"}})
    assert isinstance(c, LocalEmbeddingClient)
    # none -> disabled
    assert create_embedding_client({"llm": {"embedding_backend": "none"}}) is None


def test_local_embedding_client_default_model_and_lazy():
    c = LocalEmbeddingClient({"llm": {}})
    assert c.model == "BAAI/bge-small-en-v1.5"
    assert c._embedder is None  # not loaded until first embed()
    c2 = LocalEmbeddingClient({"llm": {"local_embedding_model": "intfloat/e5-small"}})
    assert c2.model == "intfloat/e5-small"


class FakeLLMClient(BaseLLMClient):
    """Concrete subclass for testing the abstract base."""

    def __init__(self, responses=None, config=None):
        super().__init__(config or {"llm": {}, "build": {"budget": 10.0}})
        self._responses = list(responses or [])
        self._call_count = 0

    def _call(self, prompt, system_prompt):
        self._call_count += 1
        if self._responses:
            item = self._responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return LLMResponse(content="default response", usage=LLMUsage())


class FakeStreamClient(BaseLLMClient):
    """Streamable fake: yields chunks, then returns a usage record."""

    def __init__(self, chunks=None, usage=None, setup_error=None, config=None):
        super().__init__(config or {"llm": {}, "build": {"budget": 10.0}})
        self.model = "fake/model"
        self._chunks = list(chunks or [])
        self._usage = usage
        self._setup_error = setup_error
        self.closed = False

    def _call(self, prompt, system_prompt):
        return LLMResponse(content="".join(self._chunks))

    def _call_stream(self, prompt, system_prompt):
        if self._setup_error is not None:
            raise self._setup_error
        try:
            for chunk in self._chunks:
                yield chunk
        except GeneratorExit:
            self.closed = True
            raise
        return self._usage


def _make_failover(primary, fallbacks):
    """Build a FailoverLLMClient without touching real providers."""
    fo = FailoverLLMClient.__new__(FailoverLLMClient)
    BaseLLMClient.__init__(fo, {"llm": {}, "build": {"budget": 10.0}})
    fo.primary = primary
    fo.failover_clients = list(fallbacks)
    return fo


def test_generate_stream_returns_full_content():
    client = FakeStreamClient(chunks=["hel", "lo"])
    resp = client.generate_stream("prompt")
    assert resp.content == "hello"
    assert resp.finish_reason == "stop"


def test_generate_stream_tracks_reported_usage():
    usage = LLMUsage(
        prompt_tokens=10, completion_tokens=5, total_tokens=15, estimated_cost=0.5
    )
    client = FakeStreamClient(chunks=["a", "b"], usage=usage)
    resp = client.generate_stream("prompt")
    assert resp.usage.estimated_cost == pytest.approx(0.5)
    assert client.cumulative_usage.call_count == 1
    assert client.cumulative_usage.total_tokens == 15


def test_generate_stream_estimates_usage_when_absent():
    client = FakeStreamClient(chunks=["x" * 40], usage=None)
    resp = client.generate_stream("p" * 20, "s" * 20)
    # No API usage -> estimated at ~4 chars/token, still tracked for budget.
    assert resp.usage.completion_tokens == 10
    assert resp.usage.prompt_tokens == 10
    assert client.cumulative_usage.call_count == 1


def test_generate_stream_aborts_and_closes_stream():
    client = FakeStreamClient(chunks=["good ", "bad ", "more"])
    resp = client.generate_stream("prompt", abort_check=lambda acc: "bad" in acc)
    assert resp.finish_reason == "aborted"
    assert resp.content == "good bad "
    assert client.closed is True
    # Aborted streams have no API usage; cost is still estimated and tracked.
    assert client.cumulative_usage.call_count == 1


def test_generate_stream_enforces_global_budget():
    client = FakeStreamClient(
        chunks=["x"], config={"llm": {}, "build": {"budget": 0.01}}
    )
    client.cumulative_usage.estimated_cost = 0.02
    with pytest.raises(BudgetExceededError):
        client.generate_stream("prompt")


def test_generate_stream_enforces_per_task_cap():
    client = FakeStreamClient(
        chunks=["x"],
        config={
            "llm": {},
            "build": {"budget": 100.0},
            "orchestrator": {"max_cost_per_task": 0.05},
        },
    )
    client.cost_by_task = {"T-1": 0.06}
    with client.track_task("T-1"):
        with pytest.raises(BudgetExceededError, match="Per-task budget"):
            client.generate_stream("prompt")


def test_generate_stream_unsupported_raises():
    client = FakeLLMClient([LLMResponse(content="x")])
    with pytest.raises(NotImplementedError):
        client.generate_stream("prompt")


def test_code_gen_abort_check_trips_on_filler():
    assert code_gen_abort_check("I'll help you write that code")
    assert code_gen_abort_check("Sure, here is the file")


def test_code_gen_abort_check_trips_on_long_prose_without_code():
    assert code_gen_abort_check("word " * 500)


def test_code_gen_abort_check_passes_on_code():
    assert not code_gen_abort_check("```python\nprint('hi')\n```")
    assert not code_gen_abort_check("# File: a.py\nx = 1")


def test_failover_stream_switches_on_retryable_error():
    primary = FakeStreamClient(setup_error=LLMCallError("503", retryable=True))
    backup = FakeStreamClient(
        chunks=["a", "b"], usage=LLMUsage(total_tokens=3, estimated_cost=0.2)
    )
    fo = _make_failover(primary, [backup])
    resp = fo.generate_stream("prompt")
    assert resp.content == "ab"
    assert resp.usage.estimated_cost == pytest.approx(0.2)


def test_failover_stream_propagates_non_retryable():
    primary = FakeStreamClient(setup_error=LLMCallError("bad request", retryable=False))
    backup = FakeStreamClient(chunks=["a"])
    fo = _make_failover(primary, [backup])
    with pytest.raises(LLMCallError):
        fo.generate_stream("prompt")


def _fake_openai_capture(captured):
    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content="ok"),
                        finish_reason="stop",
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    return FakeOpenAI


def _build_openrouter(monkeypatch, captured, llm=None, catalog_models=None):
    from my_project_orchestrator.core.model_catalog import ModelCatalog
    from my_project_orchestrator.llm.client import OpenRouterLLMClient

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr("openai.OpenAI", _fake_openai_capture(captured))
    config = {
        "llm": {"provider": "openrouter", **(llm or {})},
        "build": {"budget": 10.0},
    }
    client = OpenRouterLLMClient(config)
    # Inject the catalog so tests never hit the network.
    client._catalog = ModelCatalog(fetcher=lambda: list(catalog_models or []))
    return client


def test_openrouter_denies_training_providers_by_default(monkeypatch):
    captured = {}
    client = _build_openrouter(monkeypatch, captured)
    client._call("hi", "sys")
    assert captured["extra_body"] == {"provider": {"data_collection": "deny"}}


def test_openrouter_allows_training_providers_when_opted_in(monkeypatch):
    captured = {}
    client = _build_openrouter(
        monkeypatch, captured, llm={"allow_training_models": True}
    )
    client._call("hi", "sys")
    assert captured["extra_body"] == {"provider": {"data_collection": "allow"}}


def test_sampling_params_filtered_by_model_support(monkeypatch):
    captured = {}
    client = _build_openrouter(
        monkeypatch,
        captured,
        llm={
            "model": "vendor/m",
            "temperature": 0.2,
            "sampling": {"top_p": 0.9, "top_k": 40, "repetition_penalty": 1.1},
        },
        catalog_models=[
            {
                "id": "vendor/m",
                "supported_parameters": ["temperature", "top_p"],
            }
        ],
    )
    client._call("hi", "sys")
    # Only the supported knobs are sent; top_k / repetition_penalty are dropped.
    assert captured["temperature"] == 0.2
    assert captured["top_p"] == 0.9
    assert "top_k" not in captured
    assert "repetition_penalty" not in captured


def test_temperature_omitted_when_model_rejects_it(monkeypatch):
    captured = {}
    client = _build_openrouter(
        monkeypatch,
        captured,
        llm={"model": "vendor/reasoner"},
        catalog_models=[
            {
                "id": "vendor/reasoner",
                "supported_parameters": ["max_tokens", "reasoning"],
            }
        ],
    )
    client._call("hi", "sys")
    # A reasoning model that does not accept temperature must not receive it.
    assert "temperature" not in captured


def test_unknown_model_falls_back_to_temperature_only(monkeypatch):
    captured = {}
    client = _build_openrouter(
        monkeypatch,
        captured,
        llm={"model": "vendor/unlisted", "sampling": {"top_p": 0.5}},
        catalog_models=[],  # model not in catalog
    )
    client._call("hi", "sys")
    assert "temperature" in captured
    assert "top_p" not in captured


def _fake_openai_tool(captured, edits):
    """Fake OpenAI client whose completion returns an apply_edits tool call."""

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            tool_call = types.SimpleNamespace(
                function=types.SimpleNamespace(
                    name="apply_edits", arguments=json.dumps({"edits": edits})
                )
            )
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content=None, tool_calls=[tool_call]
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    return FakeOpenAI


def test_generate_edits_uses_tool_and_renders_fences(monkeypatch):
    from my_project_orchestrator.core.model_catalog import ModelCatalog
    from my_project_orchestrator.llm.client import OpenRouterLLMClient

    captured = {}
    edits = [{"path": "src/a.py", "content": "x = 1"}]
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr("openai.OpenAI", _fake_openai_tool(captured, edits))
    client = OpenRouterLLMClient(
        {
            "llm": {"provider": "openrouter", "model": "vendor/m"},
            "build": {"budget": 10.0},
        }
    )
    client._catalog = ModelCatalog(
        fetcher=lambda: [{"id": "vendor/m", "supported_parameters": ["tools"]}]
    )
    resp = client.generate_edits("do it", "sys")
    # The forced tool call is rendered back into the canonical fence format.
    assert "```text:src/a.py" in resp.content
    assert "x = 1" in resp.content
    # And the request actually forced the apply_edits tool.
    assert captured["tool_choice"]["function"]["name"] == "apply_edits"


def test_generate_edits_falls_back_when_tools_unsupported(monkeypatch):
    from my_project_orchestrator.core.model_catalog import ModelCatalog
    from my_project_orchestrator.llm.client import OpenRouterLLMClient

    captured = {}
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr("openai.OpenAI", _fake_openai_capture(captured))
    client = OpenRouterLLMClient(
        {
            "llm": {"provider": "openrouter", "model": "vendor/plain"},
            "build": {"budget": 10.0},
        }
    )
    client._catalog = ModelCatalog(
        fetcher=lambda: [
            {"id": "vendor/plain", "supported_parameters": ["temperature"]}
        ]
    )
    resp = client.generate_edits("do it", "sys")
    # No tool support -> plain generation, no tools sent.
    assert resp.content == "ok"
    assert "tools" not in captured


def test_reasoning_effort_sent_to_reasoning_model(monkeypatch):
    captured = {}
    client = _build_openrouter(
        monkeypatch,
        captured,
        llm={"model": "vendor/reasoner"},
        catalog_models=[
            {"id": "vendor/reasoner", "supported_parameters": ["reasoning"]}
        ],
    )
    with client.with_reasoning_effort("high"):
        client._call("hi", "sys")
    assert captured["extra_body"]["reasoning"] == {"effort": "high"}


def test_reasoning_effort_ignored_for_non_reasoning_model(monkeypatch):
    captured = {}
    client = _build_openrouter(
        monkeypatch,
        captured,
        llm={"model": "vendor/plain"},
        catalog_models=[
            {"id": "vendor/plain", "supported_parameters": ["temperature"]}
        ],
    )
    with client.with_reasoning_effort("high"):
        client._call("hi", "sys")
    assert "reasoning" not in captured["extra_body"]


def test_no_reasoning_when_effort_unset(monkeypatch):
    captured = {}
    client = _build_openrouter(
        monkeypatch,
        captured,
        llm={"model": "vendor/reasoner"},
        catalog_models=[
            {"id": "vendor/reasoner", "supported_parameters": ["reasoning"]}
        ],
    )
    client._call("hi", "sys")  # no with_reasoning_effort
    assert "reasoning" not in captured["extra_body"]


def test_embedding_client_embeds_in_order_and_denies_training(monkeypatch):
    from my_project_orchestrator.llm.client import OpenRouterEmbeddingClient

    captured = {}

    class FakeEmbeddings:
        def create(self, **kwargs):
            captured.update(kwargs)
            # Returned out of index order to verify the client sorts them.
            return types.SimpleNamespace(
                data=[
                    types.SimpleNamespace(index=1, embedding=[0.1, 0.2]),
                    types.SimpleNamespace(index=0, embedding=[0.3, 0.4]),
                ]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.embeddings = FakeEmbeddings()

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    client = OpenRouterEmbeddingClient(
        {"llm": {"provider": "openrouter"}, "build": {"budget": 10.0}}, "free/q"
    )
    vecs = client.embed(["t0", "t1"])
    assert vecs == [[0.3, 0.4], [0.1, 0.2]]  # reordered to input order by index
    assert captured["model"] == "free/q"
    assert captured["extra_body"] == {"provider": {"data_collection": "deny"}}
    assert "dimensions" not in captured  # default 0 -> omitted


def test_edits_to_markdown_and_extraction():
    from my_project_orchestrator.llm.client import _edits_to_markdown

    md = _edits_to_markdown([{"path": "a.py", "content": "y = 2"}, {"bad": "x"}])
    assert "```text:a.py" in md and "y = 2" in md
    # The malformed entry (no path) is skipped.
    assert md.count("```text:") == 1


def test_generate_returns_response():
    client = FakeLLMClient([LLMResponse(content="hello")])
    resp = client.generate("prompt")
    assert resp.content == "hello"


def test_generate_code_returns_string():
    client = FakeLLMClient([LLMResponse(content="code here")])
    assert client.generate_code("prompt") == "code here"


def test_usage_tracking():
    usage = LLMUsage(
        prompt_tokens=100, completion_tokens=50, total_tokens=150, estimated_cost=0.01
    )
    client = FakeLLMClient(
        [
            LLMResponse(content="r1", usage=usage),
            LLMResponse(content="r2", usage=usage),
        ]
    )
    client.generate("p1")
    client.generate("p2")
    assert client.cumulative_usage.call_count == 2
    assert client.cumulative_usage.total_tokens == 300
    assert client.cumulative_usage.estimated_cost == pytest.approx(0.02)


def test_budget_exceeded():
    client = FakeLLMClient(config={"llm": {}, "build": {"budget": 0.01}})
    client.cumulative_usage.estimated_cost = 0.02
    with pytest.raises(BudgetExceededError):
        client.generate("prompt")


def test_budget_remaining():
    client = FakeLLMClient(config={"llm": {}, "build": {"budget": 5.0}})
    client.cumulative_usage.estimated_cost = 1.5
    assert client.budget_remaining == pytest.approx(3.5)


def test_budget_remaining_floor():
    client = FakeLLMClient(config={"llm": {}, "build": {"budget": 1.0}})
    client.cumulative_usage.estimated_cost = 5.0
    assert client.budget_remaining == 0.0


def test_retry_on_retryable_error():
    client = FakeLLMClient(
        [
            LLMCallError("rate limit", retryable=True),
            LLMResponse(content="recovered"),
        ]
    )
    resp = client.generate("prompt")
    assert resp.content == "recovered"
    assert client._call_count == 2


def test_no_retry_on_non_retryable():
    client = FakeLLMClient(
        [
            LLMCallError("invalid request", retryable=False),
        ]
    )
    with pytest.raises(LLMCallError):
        client.generate("prompt")
    assert client._call_count == 1


def test_max_retries_exhausted():
    client = FakeLLMClient(
        [
            LLMCallError("timeout", retryable=True),
            LLMCallError("timeout", retryable=True),
            LLMCallError("timeout", retryable=True),
        ]
    )
    with pytest.raises(LLMCallError):
        client.generate("prompt")
    assert client._call_count == 3


def test_llm_usage_defaults():
    u = LLMUsage()
    assert u.prompt_tokens == 0
    assert u.estimated_cost == 0.0
    assert u.call_count == 0


def test_llm_response_defaults():
    r = LLMResponse(content="test")
    assert r.model == ""
    assert r.finish_reason == ""


def test_llm_call_error_retryable():
    e = LLMCallError("rate limit", retryable=True)
    assert e.retryable is True
    assert "rate limit" in str(e)


def test_llm_call_error_not_retryable():
    e = LLMCallError("bad request")
    assert e.retryable is False


def test_create_llm_client_invalid_provider():
    with pytest.raises(ValueError, match="Unsupported"):
        create_llm_client({"llm": {"provider": "nonexistent"}})


def test_per_task_cost_cap_auto_by_default():
    # Hardened default is "auto" (budget-driven cap on), per the schema — not
    # disabled. (Production configs are merged with DEFAULT_CONFIG, so this was
    # already the effective default; only unmerged test dicts saw None.)
    client = FakeLLMClient(config={"llm": {}, "build": {"budget": 10.0}})
    assert client._max_cost_per_task == "auto"


def test_per_task_cost_cap_disabled_when_explicitly_none():
    client = FakeLLMClient(
        config={
            "llm": {},
            "build": {"budget": 10.0},
            "orchestrator": {"max_cost_per_task": None},
        }
    )
    assert client._max_cost_per_task is None
    assert not client.task_cost_exceeded("T-1")


def test_per_task_cost_cap_stops_task_generation():
    client = FakeLLMClient(
        config={
            "llm": {},
            "build": {"budget": 100.0},
            "orchestrator": {"max_cost_per_task": 0.05},
        }
    )
    # Simulate prior spend attributed to this task above the cap.
    client.cost_by_task = {"T-1": 0.06}
    assert client.task_cost("T-1") == pytest.approx(0.06)
    assert client.task_cost_exceeded("T-1")
    with client.track_task("T-1"):
        with pytest.raises(BudgetExceededError, match="Per-task budget"):
            client.generate("prompt")
    # A different task under the cap is unaffected.
    with client.track_task("T-2"):
        assert client.generate("prompt").content == "default response"


def test_per_task_cost_cap_auto_is_budget_fraction():
    client = FakeLLMClient(
        config={
            "llm": {},
            "build": {"budget": 10.0},
            "orchestrator": {"max_cost_per_task": "auto"},
        }
    )
    # "auto" snapshots half the budget remaining when the task starts.
    with client.track_task("T-1"):
        cap = client.effective_task_cap("T-1")
    assert cap == pytest.approx(5.0)
    client.cost_by_task = {"T-1": 4.0}
    assert not client.task_cost_exceeded("T-1")
    client.cost_by_task = {"T-1": 6.0}
    assert client.task_cost_exceeded("T-1")
