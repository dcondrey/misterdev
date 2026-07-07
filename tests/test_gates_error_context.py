from misterdev.task_executors.markdown_plan_executor.gates_mixin import (
    GatesMixin,
    _AttemptFailure,
    _error_signature,
)


def test_error_signature_folds_out_line_numbers():
    # The SAME error at different source lines must fingerprint identically, so a
    # fix that only shifted line numbers still counts as "the same wall".
    a = _error_signature("error[E0425]: cannot find value `x` in this scope, line 10")
    b = _error_signature("error[E0425]: cannot find value `x` in this scope, line 42")
    assert a == b


def test_error_signature_differs_for_different_errors():
    a = _error_signature("error[E0425]: cannot find value `x` in this scope")
    b = _error_signature("error[E0308]: mismatched types: expected u32, found String")
    assert a != b


def test_repeat_escalation_empty_below_two():
    assert GatesMixin._repeat_escalation([]) == ""
    assert GatesMixin._repeat_escalation([_AttemptFailure("a", "sig")]) == ""


def test_repeat_escalation_silent_when_errors_differ():
    prior = [_AttemptFailure("a", "sig1"), _AttemptFailure("b", "sig2")]
    assert GatesMixin._repeat_escalation(prior) == ""


def test_repeat_escalation_fires_on_repeated_signature():
    prior = [_AttemptFailure("a", "sig1"), _AttemptFailure("b", "sig1")]
    banner = GatesMixin._repeat_escalation(prior)
    assert "CHANGE YOUR APPROACH" in banner
    assert "2x" in banner


def test_build_error_context_prepends_escalation_on_identical_recurrence():
    # Two attempts hit the same underlying error (only the line number moved).
    # The second must carry the "change your approach" banner; the first must not.
    gates = GatesMixin()
    prior: list = []
    err1 = "error[E0425]: cannot find value `foo` in this scope, at line 10"
    err2 = "error[E0425]: cannot find value `foo` in this scope, at line 55"

    ctx1 = gates._build_error_context(prior, 0, err1, "CLASSIFIED", "AT file.rs:10")
    assert "CHANGE YOUR APPROACH" not in ctx1

    ctx2 = gates._build_error_context(prior, 1, err2, "CLASSIFIED", "AT file.rs:55")
    assert "CHANGE YOUR APPROACH" in ctx2
    # The original per-category guidance and attributed error are still present:
    # escalation is additive, it never replaces the existing context.
    assert "CLASSIFIED" in ctx2 and "AT file.rs:55" in ctx2


def test_build_error_context_no_escalation_when_errors_differ():
    gates = GatesMixin()
    prior: list = []
    gates._build_error_context(
        prior, 0, "error[E0425]: cannot find value `a`", "C", "L"
    )
    ctx2 = gates._build_error_context(
        prior, 1, "error[E0308]: mismatched types here", "C", "L"
    )
    assert "CHANGE YOUR APPROACH" not in ctx2
