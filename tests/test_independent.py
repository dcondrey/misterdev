from contextlib import contextmanager

from misterdev.core.verification.independent import (
    generate_independent,
    build_independent_call,
)


class _Client:
    def __init__(self):
        self.active = None
        self.seen = None

    @contextmanager
    def with_model(self, model):
        self.active = model
        try:
            yield self
        finally:
            self.active = None

    def generate_code(self, prompt, system=""):
        self.seen = self.active
        return "OUT"


class _NoSwitchClient:
    def generate_code(self, prompt, system=""):
        return "OUT"


def test_generate_independent_uses_model():
    c = _Client()
    assert generate_independent(c, "p", "s", model="other/m") == "OUT"
    assert c.seen == "other/m"


def test_generate_independent_without_model_uses_generator():
    c = _Client()
    generate_independent(c, "p", model=None)
    assert c.seen is None


def test_generate_independent_client_without_with_model():
    assert generate_independent(_NoSwitchClient(), "p", model="other/m") == "OUT"


def test_generate_independent_coerces_none_to_empty_string():
    class _NoneClient:
        def generate_code(self, prompt, system=""):
            return None

    assert generate_independent(_NoneClient(), "p") == ""


def test_build_independent_call_none_without_client():
    assert build_independent_call(None, "sys", "m", "role") is None


def test_build_independent_call_none_without_generate_code():
    assert build_independent_call(object(), "sys", "m", "role") is None


def test_build_independent_call_routes_through_model():
    c = _Client()
    call = build_independent_call(c, "sys", "ind/model", "Critic")
    assert call("prompt") == "OUT"
    assert c.seen == "ind/model"


def test_build_independent_call_falls_back_when_no_switch():
    # critic.model set but client can't switch -> runs on own model, no crash.
    call = build_independent_call(_NoSwitchClient(), "sys", "ind/model", "Critic")
    assert call("prompt") == "OUT"


def test_build_independent_call_no_model_uses_generator():
    c = _Client()
    call = build_independent_call(c, "sys", None, "Judge")
    call("prompt")
    assert c.seen is None
