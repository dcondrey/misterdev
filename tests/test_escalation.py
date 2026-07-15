"""Escalation ladder: rung selection and the infra-never-advances guarantee."""

from misterdev.core.execution.escalation import choose_rung, should_count_failure


def test_ladder_climbs_with_code_failures():
    assert choose_rung(0) == "normal"
    assert choose_rung(1) == "widen_context"
    assert choose_rung(2) == "stronger_model"
    assert choose_rung(3) == "decompose"
    assert choose_rung(9) == "decompose"  # decompose is terminal


def test_custom_thresholds():
    kw = dict(widen_after=2, model_after=4, decompose_after=6)
    assert choose_rung(1, **kw) == "normal"
    assert choose_rung(2, **kw) == "widen_context"
    assert choose_rung(4, **kw) == "stronger_model"
    assert choose_rung(6, **kw) == "decompose"


def test_misordered_thresholds_stay_monotonic():
    # Highest threshold met wins, so a nonsensical config still yields a defined
    # (never lower-than-expected) rung.
    assert (
        choose_rung(5, widen_after=3, model_after=1, decompose_after=2) == "decompose"
    )


def test_should_count_failure_ignores_infra():
    for infra in (
        "Command timed out after 120s",
        "waiting for the lock on the store",
        "ENOSPC: no space left on device",
        "JavaScript heap out of memory",
    ):
        assert should_count_failure(infra) is False


def test_should_count_failure_counts_real_code_errors():
    for code in (
        "error TS2345: Argument of type 'string' is not assignable",
        "AssertionError: expected 3 got 4",
        "SyntaxError: Unexpected token )",
    ):
        assert should_count_failure(code) is True


def test_infra_failures_never_advance_the_ladder():
    """A counter that only increments on should_count_failure escalates on CODE
    failures and holds across interleaved INFRA failures."""
    outputs = [
        "Command timed out after 120s",  # infra  -> hold at normal
        "error TS2345: bad type",  # code   -> widen
        "ELOCK: store is locked",  # infra  -> hold at widen
        "AssertionError: 1 != 2",  # code   -> stronger_model
        "operation timed out",  # infra  -> hold at stronger_model
        "error TS2304: name not found",  # code   -> decompose
    ]
    code_failures = 0
    rungs = []
    for out in outputs:
        if should_count_failure(out):
            code_failures += 1
        rungs.append(choose_rung(code_failures))
    assert rungs == [
        "normal",
        "widen_context",
        "widen_context",
        "stronger_model",
        "stronger_model",
        "decompose",
    ]
