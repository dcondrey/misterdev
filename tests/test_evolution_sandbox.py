from dataclasses import dataclass, field

import pytest

from misterdev.core.evolution import (
    Mutation,
    ProtectedPathError,
    SandboxEvaluator,
)


@dataclass
class _R:
    resolved: bool
    name: str = "ex"


@dataclass
class _Report:
    resolved: int
    total: int
    results: list = field(default_factory=list)


def _mut(paths=("misterdev/config.py",)):
    return Mutation(target="rust", paths=list(paths), patch="diff")


def test_gate_failure_returns_none_and_skips_benchmark():
    ran = {"benchmark": False}

    def benchmark():
        ran["benchmark"] = True
        return _Report(9, 10), 5.0

    ev = SandboxEvaluator(
        apply=lambda m: lambda: None,
        gates=lambda: False,
        benchmark=benchmark,
    )
    assert ev(_mut()) is None
    assert ran["benchmark"] is False  # never spent a cent on a broken build


def test_scores_and_counts_regressions_against_baseline():
    report = _Report(
        resolved=2, total=3, results=[_R(True, "a"), _R(True, "b"), _R(False, "c")]
    )
    ev = SandboxEvaluator(
        apply=lambda m: lambda: None,
        gates=lambda: True,
        benchmark=lambda: (report, 3.0),
        baseline_passed={
            "a",
            "b",
            "c",
        },  # c passed at baseline, fails now -> regression
    )
    score = ev(_mut())
    assert (score.resolved, score.total, score.cost) == (2, 3, 3.0)
    assert score.regressions == 1


def test_no_regression_when_new_passes_cover_baseline():
    report = _Report(2, 2, results=[_R(True, "a"), _R(True, "b")])
    ev = SandboxEvaluator(
        apply=lambda m: lambda: None,
        gates=lambda: True,
        benchmark=lambda: (report, 1.0),
        baseline_passed={"a"},
    )
    assert ev(_mut()).regressions == 0


def test_teardown_runs_on_gate_failure_and_on_benchmark_raise():
    torn = {"count": 0}

    def apply(m):
        return lambda: torn.__setitem__("count", torn["count"] + 1)

    # gate failure path
    SandboxEvaluator(
        apply=apply, gates=lambda: False, benchmark=lambda: (_Report(1, 1), 0.0)
    )(_mut())

    # benchmark raises path
    def boom():
        raise RuntimeError("benchmark exploded")

    with pytest.raises(RuntimeError):
        SandboxEvaluator(apply=apply, gates=lambda: True, benchmark=boom)(_mut())
    assert torn["count"] == 2  # worktree cleaned up on both paths


def test_guardrail_refuses_protected_target_before_applying():
    applied = {"count": 0}

    def apply(m):
        applied["count"] += 1
        return lambda: None

    ev = SandboxEvaluator(
        apply=apply, gates=lambda: True, benchmark=lambda: (_Report(1, 1), 0.0)
    )
    with pytest.raises(ProtectedPathError):
        ev(_mut(paths=["evaluation/polyglot/grader.py"]))
    assert applied["count"] == 0  # never touched the worktree for a walled-off edit
