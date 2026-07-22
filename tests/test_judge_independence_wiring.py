"""R3 — the acceptance judge honors the same-model independence check (T1.2 residual).

`GatesMixin._judge_generate` (the LLM PASS/FAIL acceptance judge — the highest-stakes
judge) called `generate_independent` directly, which routes through `with_model`
whenever a model is set. When `judge.model` equals the generator's model that is false
independence. Routing through the shared `build_independent_call` makes it detect and
skip the same-model case here too.
"""

import contextlib
from types import SimpleNamespace

from misterdev.task_executors.markdown_plan_executor.gates_mixin import GatesMixin


class _Ex(GatesMixin):
    pass


class _Switchable:
    def __init__(self, model="gen-model"):
        self._model = model
        self.entered = []

    @property
    def model(self):
        return self._model

    def generate_code(self, prompt, system=""):
        return "PASS"

    @contextlib.contextmanager
    def with_model(self, m):
        self.entered.append(m)
        prev, self._model = self._model, m
        try:
            yield
        finally:
            self._model = prev


def _project(judge_model, client):
    return SimpleNamespace(llm_client=client, config={"judge": {"model": judge_model}})


def test_same_model_judge_is_not_routed():
    client = _Switchable("gen-model")
    _Ex()._judge_generate(_project("gen-model", client), "prompt")
    assert client.entered == []


def test_distinct_model_judge_routes():
    client = _Switchable("gen-model")
    out = _Ex()._judge_generate(_project("other-model", client), "prompt")
    assert client.entered == ["other-model"]
    assert out == "PASS"


def test_client_without_generate_code_returns_empty():
    project = SimpleNamespace(llm_client=object(), config={"judge": {}})
    assert _Ex()._judge_generate(project, "prompt") == ""
