"""T1.2 — the judge/critic must run on a model INDEPENDENT of the generator.

`build_independent_call` already warns when a model is set but the client can't
switch, and when no model is configured. The silent hole: when the configured judge
model EQUALS the generator's current model, it claims independence and routes to the
same model — zero independence, no signal. This asserts that same-model is detected
(warned) and NOT treated as independent routing, while a genuinely different model
still routes.
"""

import contextlib

from misterdev.core.verification.independent import build_independent_call


class _Switchable:
    def __init__(self, model="gen-model"):
        self._model = model
        self.entered = []

    @property
    def model(self):
        return self._model

    def generate_code(self, prompt, system=""):
        return "verdict"

    @contextlib.contextmanager
    def with_model(self, m):
        self.entered.append(m)
        prev, self._model = self._model, m
        try:
            yield
        finally:
            self._model = prev


class _NoSwitch:
    model = "gen-model"

    def generate_code(self, prompt, system=""):
        return "verdict"


def test_same_model_is_not_routed_as_independent(caplog):
    client = _Switchable("gen-model")
    call = build_independent_call(client, "", "gen-model", "Judge")
    assert call is not None
    call("prompt")
    # Same model as the generator -> not real independence -> must NOT route through
    # with_model (which would falsely imply an independent check).
    assert client.entered == []
    assert any("same" in r.message.lower() for r in caplog.records)


def test_distinct_model_routes_independently():
    client = _Switchable("gen-model")
    call = build_independent_call(client, "", "other-model", "Judge")
    call("prompt")
    assert client.entered == ["other-model"]


def test_no_model_runs_on_generator_without_routing():
    client = _Switchable("gen-model")
    call = build_independent_call(client, "", None, "Judge")
    call("prompt")
    assert client.entered == []


def test_unroutable_model_still_warns_and_runs():
    client = _NoSwitch()
    call = build_independent_call(client, "", "other-model", "Judge")
    assert call is not None
    assert call("prompt") == "verdict"
