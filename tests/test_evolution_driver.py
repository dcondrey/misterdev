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


def test_real_failure_target_overrides_and_bypasses_short_circuit():
    from misterdev.core.evolution.attribution import Blame

    # Benchmark is all-green, so benchmark blame is None and the run would normally
    # short-circuit. A real-failure target drives a proposal anyway — evolution
    # improves what actually breaks in use, not only the benchmark.
    results = [BenchResult("a", "rust", True), BenchResult("b", "go", True)]
    target = Blame(
        niche="rust/wrong_type",
        failures=4,
        total=4,
        examples=["E0308"],
        source="real-build failures",
    )
    mutation = Mutation(
        target="rust/wrong_type", paths=["misterdev/x.py"], patch="p", note="prompt"
    )
    res = run_evolution(
        _Project(),
        "bench",
        "work",
        run_bench=_bench(results),
        proposer=_Proposer(mutation),
        live=False,
        target=target,
    )
    assert res.blame is target
    assert res.blame.source == "real-build failures"
    assert res.proposals == [mutation]


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


class _ScreeningSandbox(_Sandbox):
    """Sandbox that can run a case subset (benchmark_only), so the screen arms."""

    def __init__(self, improved):
        super().__init__(improved)
        self.only_calls = []

    def benchmark_only(self, only):
        self.only_calls.append(list(only))
        # The mutation flips every requested case to passing.
        return [BenchResult(cid, "rust", True) for cid in only], 0.001


def test_live_run_arms_screen_from_corpus_and_promotes(tmp_path):
    # Baseline: a,b fail (targets), g passes (guard). The screen should run the
    # targeted+guard subset, accept, and the oracle should then promote.
    baseline = [
        BenchResult("a", "rust", False, "error[E0308]: mismatched types"),
        BenchResult("b", "rust", False, "error[E0308]: mismatched types"),
        BenchResult("g", "rust", True),
    ]
    improved = [BenchResult(x, "rust", True) for x in ("a", "b", "g")]
    mutation = Mutation(
        target="rust", paths=["misterdev/x.py"], patch="p", note="prompt"
    )
    sandbox = _ScreeningSandbox(improved)
    res = run_evolution(
        _Project(),
        "bench",
        "work",
        run_bench=_bench(baseline),
        proposer=_Proposer(mutation),
        sandbox=sandbox,
        live=True,
        steps=1,
        noise_band=0.05,
        screen=True,
        beam=3,
        corpus_path=tmp_path / "repro.json",
    )
    assert sandbox.only_calls, "screen never ran the targeted subset"
    # The subset ran the targets and the guard, not the whole suite.
    assert set(sandbox.only_calls[0]) == {"a", "b", "g"}
    assert res.steps[0].promoted


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
