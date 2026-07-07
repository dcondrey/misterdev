import tempfile
from pathlib import Path

from misterdev.core.evolution.adapters import BenchResult
from misterdev.core.evolution.driver import run_evolution
from misterdev.core.evolution.loop import Mutation


class _Project:
    def __init__(self):
        self.path = Path(tempfile.mkdtemp())


class _Proposer:
    def __init__(self, mutation):
        self._m = mutation

    def propose(self, blame, favored_kinds=None):
        return self._m


def _bench(results, cost=0.0):
    return lambda cwd: (results, cost, {"results": results})


def test_dry_run_reports_baseline_blame_and_a_proposal_without_applying():
    results = [
        BenchResult("a", "rust", False, "error[E0308]"),
        BenchResult("b", "rust", False, "error[E0308]"),
        BenchResult("c", "go", True),
    ]
    mutation = Mutation(
        target="rust", paths=["misterdev/x.py"], patch="p", note="prompt"
    )
    res = run_evolution(
        _Project(),
        "bench",
        "work",
        run_bench=_bench(results),
        proposer=_Proposer(mutation),
        live=False,
    )
    assert res.baseline.resolved == 1 and res.baseline.total == 3
    assert res.blame.niche.startswith("rust")
    assert res.proposals == [mutation]
    assert res.steps == []  # nothing applied or promoted


def test_all_passing_baseline_short_circuits():
    results = [BenchResult("a", "rust", True), BenchResult("b", "go", True)]
    res = run_evolution(
        _Project(), "bench", "work", run_bench=_bench(results), proposer=_Proposer(None)
    )
    assert res.blame is None
    assert res.note == "nothing to improve"


class _Sandbox:
    """Fake sandbox: applies nothing, gates pass, benchmark returns an improved run."""

    def __init__(self, improved):
        self._improved = improved

    def apply(self, mutation):
        return lambda: None

    def gates(self):
        return True

    def benchmark(self):
        from misterdev.core.evolution.adapters import _DuckReport

        return _DuckReport(self._improved), 0.0


def test_live_run_promotes_a_real_improvement():
    baseline = [BenchResult("a", "rust", False), BenchResult("b", "rust", False)]
    improved = [BenchResult("a", "rust", True), BenchResult("b", "rust", True)]
    mutation = Mutation(
        target="rust", paths=["misterdev/x.py"], patch="p", note="prompt"
    )
    res = run_evolution(
        _Project(),
        "bench",
        "work",
        run_bench=_bench(baseline),
        proposer=_Proposer(mutation),
        sandbox=_Sandbox(improved),
        live=True,
        steps=1,
        noise_band=0.05,
    )
    assert len(res.steps) == 1
    assert res.steps[0].promoted
    assert res.champion is not None and res.champion.resolved == 2


def test_live_run_does_not_promote_within_noise():
    baseline = [BenchResult(f"e{i}", "rust", i < 5) for i in range(10)]  # 5/10
    barely = [
        BenchResult(f"e{i}", "rust", i < 6) for i in range(10)
    ]  # 6/10, +10% > band? no, band .2
    mutation = Mutation(
        target="rust", paths=["misterdev/x.py"], patch="p", note="prompt"
    )
    res = run_evolution(
        _Project(),
        "bench",
        "work",
        run_bench=_bench(baseline),
        proposer=_Proposer(mutation),
        sandbox=_Sandbox(barely),
        live=True,
        steps=1,
        noise_band=0.20,
    )
    assert not res.steps[0].promoted
