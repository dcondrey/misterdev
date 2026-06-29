"""Integration: the executor's model-selection / ledger / cache seams.

Exercises the wiring on a stub project carrying the real ModelLedger,
ModelSelector, and LLMCache, without standing up the full Try-Test-Fix loop.
"""

import contextlib
import tempfile
import types
from pathlib import Path

import pytest

from my_project_orchestrator.config import get_setting
from my_project_orchestrator.core.economics.llm_cache import LLMCache
from my_project_orchestrator.core.economics.model_ledger import ModelLedger
from my_project_orchestrator.core.economics.model_selector import ModelSelector
from my_project_orchestrator.task_executors.markdown_plan_executor import (
    MarkdownPlanExecutor,
)


class StubLLMClient:
    def __init__(self):
        self.model = "default/model"
        self.calls = 0
        self._cost = {}

    def task_cost(self, task_id):
        return self._cost.get(task_id, 0.0)

    def generate(self, prompt, system_prompt=""):
        from my_project_orchestrator.llm.client import LLMResponse

        return LLMResponse(content=self.generate_code(prompt, system_prompt))

    def generate_code(self, prompt, system_prompt):
        self.calls += 1
        return "FRESH OUTPUT"


class RoutingStubClient:
    """Client that can fail for specific routed models, to test fallback."""

    def __init__(self, fail_models=()):
        self.model = "default/model"
        self.fail_models = set(fail_models)
        self.calls = []
        self._cost = {}

    def task_cost(self, task_id):
        return self._cost.get(task_id, 0.0)

    @contextlib.contextmanager
    def with_model(self, model):
        prev = self.model
        self.model = model
        try:
            yield
        finally:
            self.model = prev

    def generate(self, prompt, system_prompt=""):
        from my_project_orchestrator.llm.client import LLMResponse

        return LLMResponse(content=self.generate_code(prompt, system_prompt))

    def generate_code(self, prompt, system_prompt):
        self.calls.append(self.model)
        if self.model in self.fail_models:
            raise RuntimeError(f"no permitted provider for {self.model}")
        return f"OUT::{self.model}"


class StubProject:
    def __init__(self, path, config, client=None):
        self.path = Path(path)
        self.config = config
        self.llm_client = client or StubLLMClient()
        self.model_ledger = ModelLedger(
            self.path / ".orchestrator" / "model_stats.json"
        )
        self.model_selector = ModelSelector(config, self.model_ledger)
        self.llm_cache = (
            LLMCache(self.path / ".orchestrator" / "llm_cache")
            if get_setting(config, "llm", "cache")
            else None
        )


def _task(category="feature", complexity="medium", task_id="T-1"):
    return types.SimpleNamespace(category=category, complexity=complexity, id=task_id)


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_cache_hit_skips_model_call(tmpdir):
    config = {"llm": {"cache": True}, "build": {"budget": 10.0}}
    project = StubProject(tmpdir, config)
    project.llm_cache.put("SYS", "PROMPT", "CACHED OUTPUT")
    ex = MarkdownPlanExecutor()
    out, aborted = ex._invoke_llm(project, "PROMPT", "SYS")
    assert out == "CACHED OUTPUT"
    assert aborted is False
    assert project.llm_client.calls == 0  # no model call on a hit


def test_cache_miss_calls_model(tmpdir):
    config = {"llm": {"cache": True}, "build": {"budget": 10.0}}
    project = StubProject(tmpdir, config)
    out, _ = MarkdownPlanExecutor()._invoke_llm(project, "PROMPT", "SYS")
    assert out == "FRESH OUTPUT"
    assert project.llm_client.calls == 1


def test_cache_disabled_always_calls_model(tmpdir):
    config = {"llm": {"cache": False}, "build": {"budget": 10.0}}
    project = StubProject(tmpdir, config)
    ex = MarkdownPlanExecutor()
    ex._invoke_llm(project, "PROMPT", "SYS")
    ex._invoke_llm(project, "PROMPT", "SYS")
    assert project.llm_client.calls == 2  # no caching


def test_cache_store_roundtrip(tmpdir):
    config = {"llm": {"cache": True}, "build": {"budget": 10.0}}
    project = StubProject(tmpdir, config)
    ex = MarkdownPlanExecutor()
    ex._cache_store(project, "SYS", "PROMPT", "STORED", "free/x")
    out, _ = ex._invoke_llm(project, "PROMPT", "SYS")
    assert out == "STORED"
    assert project.llm_client.calls == 0


def test_select_model_uses_proven_cheap(tmpdir):
    config = {
        "llm": {
            "dynamic_selection": True,
            "escalation": ["cheap", "strong"],
            "models": {"cheap": "free/x", "strong": "paid/big"},
            "min_observations": 2,
            "first_try_floor": 0.5,
            "selection_posture": "conservative",
        },
        "build": {"budget": 10.0},
    }
    project = StubProject(tmpdir, config)
    for _ in range(2):
        project.model_ledger.record(
            "free/x", "feature", "medium", success=True, first_try=True, cost=0.001
        )
    ex = MarkdownPlanExecutor()
    assert ex._select_model(project, _task(), "iterative", 0, 3) == "free/x"
    # Final attempt always escalates to the strongest tier.
    assert ex._select_model(project, _task(), "iterative", 2, 3) == "paid/big"


def test_select_model_disabled_falls_back_to_static(tmpdir):
    config = {"llm": {"dynamic_selection": False}, "build": {"budget": 10.0}}
    project = StubProject(tmpdir, config)
    ex = MarkdownPlanExecutor()
    # No static routing configured either -> None (client keeps its default).
    assert ex._select_model(project, _task(), "iterative", 0, 3) is None


def test_ledger_record_persists_outcome(tmpdir):
    config = {
        "llm": {
            "dynamic_selection": True,
            "escalation": ["cheap"],
            "models": {"cheap": "free/x"},
        },
        "build": {"budget": 10.0},
    }
    project = StubProject(tmpdir, config)
    ex = MarkdownPlanExecutor()
    pending = {
        "model": "free/x",
        "attempt": 0,
        "cost_before": 0.0,
        "latency": 1.5,
        "aborted": False,
    }
    ex._ledger_record(project, _task(), pending, success=True)
    s = project.model_ledger.stat("free/x", "feature", "medium")
    assert s.successes == 1
    assert s.first_try_successes == 1


def test_invoke_routed_falls_back_when_routed_model_fails(tmpdir):
    config = {
        "llm": {
            "dynamic_selection": True,
            "escalation": ["cheap", "strong"],
            "models": {"cheap": "free/x", "strong": "default/model"},
            "cache": False,
        },
        "build": {"budget": 10.0},
    }
    client = RoutingStubClient(fail_models={"free/x"})
    project = StubProject(tmpdir, config, client=client)
    ex = MarkdownPlanExecutor()
    resp, aborted, pending = ex._invoke_routed(
        project, _task(), "p", "s", "free/x", 0, True
    )
    # Free model failed -> degraded to the default model, task not aborted.
    assert resp == "OUT::default/model"
    assert pending["model"] == "default/model"
    assert client.calls == ["free/x", "default/model"]
    # The free model's availability failure is recorded so it is deprioritized.
    failed = project.model_ledger.stat("free/x", "feature", "medium")
    assert failed.attempts == 1 and failed.successes == 0


def test_invoke_routed_no_fallback_on_success(tmpdir):
    config = {
        "llm": {"dynamic_selection": True, "cache": False},
        "build": {"budget": 10.0},
    }
    client = RoutingStubClient()
    project = StubProject(tmpdir, config, client=client)
    ex = MarkdownPlanExecutor()
    resp, _, pending = ex._invoke_routed(project, _task(), "p", "s", "free/x", 0, True)
    assert resp == "OUT::free/x"
    assert pending["model"] == "free/x"
    assert client.calls == ["free/x"]


def test_reasoning_ctx_maps_complexity_to_effort(tmpdir):
    config = {
        "llm": {
            "reasoning_effort": {"large": "high", "medium": "medium"},
            "cache": False,
        },
        "build": {"budget": 10.0},
    }
    captured = {}

    class EffortClient:
        @contextlib.contextmanager
        def with_reasoning_effort(self, effort):
            captured["effort"] = effort
            yield

    project = StubProject(tmpdir, config, client=EffortClient())
    ex = MarkdownPlanExecutor()
    with ex._reasoning_ctx(project, _task(complexity="large")):
        pass
    assert captured["effort"] == "high"
    captured.clear()
    # A complexity absent from the map gets no reasoning request.
    with ex._reasoning_ctx(project, _task(complexity="small")):
        pass
    assert "effort" not in captured
