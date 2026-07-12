"""Partial-credit (ordinal) fitness: a measurement-only, lower-variance signal
that distinguishes progress binary pass/fail cannot see."""

from misterdev.core.evolution.partial_credit import (
    CHECKPOINTS,
    PartialCreditScore,
    checkpoint_credit,
    furthest_checkpoint,
)


def test_checkpoint_credit_is_ordered_and_bounded():
    credits = [checkpoint_credit(c) for c in CHECKPOINTS]
    assert credits == sorted(credits)  # monotonic along the ladder
    assert checkpoint_credit(None) == 0.0
    assert checkpoint_credit("unknown") == 0.0
    assert checkpoint_credit("suite") == 1.0  # full credit == a binary pass
    assert 0.0 < checkpoint_credit("compile") < checkpoint_credit("query") < 1.0


def test_furthest_checkpoint_from_unordered_set():
    assert furthest_checkpoint({"compile", "invariant", "construct"}) == "invariant"
    assert furthest_checkpoint([]) is None
    assert furthest_checkpoint({"nonsense"}) is None


def test_from_checkpoints_accepts_mixed_inputs():
    s = PartialCreditScore.from_checkpoints(
        ["suite", {"compile", "construct"}, None, "compile"]
    )
    assert s.total == 4
    assert s.resolved == 1  # only "suite" is full credit
    assert 0.0 < s.mean_credit < 1.0


def test_resolved_matches_binary_count():
    # Full-credit tasks equal the binary resolved count; the metric never
    # over-reports a pass.
    s = PartialCreditScore.from_checkpoints(["suite", "suite", "query", None])
    assert s.resolved == 2 and s.resolved_rate == 0.5


def test_partial_credit_sees_progress_binary_is_blind_to():
    # Two runs with the SAME binary pass-rate (0 resolved) but different real
    # progress: run B got every task to "invariant", run A only to "compile".
    a = PartialCreditScore.from_checkpoints(["compile", "compile", "compile"])
    b = PartialCreditScore.from_checkpoints(["invariant", "invariant", "invariant"])
    assert a.resolved_rate == b.resolved_rate == 0.0  # binary can't tell them apart
    assert b.mean_credit > a.mean_credit  # partial credit can
    assert b.delta(a) > 0


def test_variance_and_stderr_shrink_with_agreement():
    uniform = PartialCreditScore.from_checkpoints(["query"] * 8)
    spread = PartialCreditScore.from_checkpoints(
        ["suite", None, "suite", None, "suite", None, "suite", None]
    )
    assert uniform.variance == 0.0  # all-agree -> no spread
    assert spread.variance > uniform.variance


def test_resolves_is_advisory_and_needs_margin():
    lo = PartialCreditScore.from_checkpoints(["compile"] * 10)
    hi = PartialCreditScore.from_checkpoints(["suite"] * 10)
    assert hi.resolves(lo)  # a large, unambiguous gain resolves
    # A one-task flicker does not clear the z-margin.
    near = PartialCreditScore.from_checkpoints(["query"] * 10)
    near2 = PartialCreditScore.from_checkpoints(["query"] * 9 + ["suite"])
    assert not near2.resolves(near)


def test_empty_is_safe():
    e = PartialCreditScore.from_checkpoints([])
    assert e.mean_credit == 0.0 and e.variance == 0.0 and e.stderr == 0.0
