"""Unit tests for the failure-reflection (Reflexion) component."""

from misterdev.core.verification.reflection import (
    _MAX_REFLECTION_CHARS,
    reflect_on_failure,
)


def test_reflection_returns_model_text():
    seen = {}

    def call(prompt):
        seen["prompt"] = prompt
        return "Root cause: off-by-one. Change: start the loop at 0."

    out = reflect_on_failure(
        "Implement foo", "AssertionError: 4 != 5", reflect_call=call
    )
    assert "off-by-one" in out
    # The failing output and task are in the prompt the reflector sees.
    assert "AssertionError" in seen["prompt"]
    assert "Implement foo" in seen["prompt"]


def test_reflection_folds_in_prior_reflections():
    def call(prompt):
        # The prompt must carry earlier reflections so the model builds on them.
        assert "earlier reflection A" in prompt
        return "new reflection"

    out = reflect_on_failure(
        "task", "error", prior_reflections=["earlier reflection A"], reflect_call=call
    )
    assert out == "new reflection"


def test_reflection_skips_without_client_or_output():
    # No call and no client -> skip (empty), never blocks the retry.
    assert reflect_on_failure("task", "error", llm_client=None) == ""
    # Empty failing output -> nothing to reflect on.
    assert reflect_on_failure("task", "", reflect_call=lambda p: "x") == ""


def test_reflection_skips_on_error():
    def boom(prompt):
        raise RuntimeError("model down")

    # A reflector failure is swallowed -> "" (the retry proceeds unreflected).
    assert reflect_on_failure("task", "error", reflect_call=boom) == ""


def test_reflection_is_length_bounded():
    out = reflect_on_failure("task", "error", reflect_call=lambda p: "x" * 10000)
    assert len(out) <= _MAX_REFLECTION_CHARS
