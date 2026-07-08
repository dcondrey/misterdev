"""End-to-end (fakes, $0): the live loop's held-out gate rejects an overfit
mutation and promotes a generalizing one. Exercises the real
driver -> SandboxEvaluator -> loop -> decide_promotion path without a benchmark
or an LLM, by injecting run_bench / sandbox / proposer.
"""

from types import SimpleNamespace

from misterdev.core.evolution.driver import run_evolution
from misterdev.core.evolution.holdout import split_tasks
from misterdev.core.evolution.loop import Mutation

_SLUGS = [f"ex{i}" for i in range(12)]
_DERIVE, _HOLDOUT = split_tasks(_SLUGS, holdout_fraction=0.3)
_HOLD = set(_HOLDOUT)


def _res(name, resolved):
    # Baseline error only on failing tasks so there is a niche to blame.
    return SimpleNamespace(
        name=name,
        language="rust",
        resolved=resolved,
        error="" if resolved else "assertion `left == right` failed",
    )


# Baseline: every DERIVE task fails, every HOLDOUT task passes.
_BASELINE = [_res(s, s in _HOLD) for s in _SLUGS]


class _FakeProposer:
    def propose(self, blame, favored_kinds=None):
        return Mutation(
            target=blame.niche,
            paths=["misterdev/core/context/guidance/rust.py"],
            patch=(
                "```python:misterdev/core/context/guidance/rust.py\n"
                "<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE\n```"
            ),
            note="guidance-rule",
        )


class _FakeSandbox:
    """apply/gates/benchmark that a real SandboxEvaluator drives; benchmark returns
    a controlled mutated report so the held-out split is deterministic."""

    def __init__(self, mutated):
        self._mutated = mutated

    def apply(self, mutation):
        return lambda: None  # no worktree; teardown is a no-op

    def gates(self):
        return True  # misterdev's own build/tests "pass" in the fake

    def benchmark(self):
        rep = SimpleNamespace(
            results=self._mutated,
            resolved=sum(1 for r in self._mutated if r.resolved),
            total=len(self._mutated),
        )
        return rep, 0.01


def _run(mutated, tmp_path):
    project = SimpleNamespace(path=tmp_path, name="t", config={})
    return run_evolution(
        project,
        benchmark_dir="unused",
        workdir=str(tmp_path / "wd"),
        live=True,
        steps=1,
        noise_band=0.05,
        run_bench=lambda cwd: (_BASELINE, 0.0, {}),
        sandbox=_FakeSandbox(mutated),
        proposer=_FakeProposer(),
    )


def test_overfit_mutation_is_rejected_by_holdout_gate(tmp_path):
    # Mutated: DERIVE now all pass (gain) but HOLDOUT now all fail (dropped).
    mutated = [_res(s, s not in _HOLD) for s in _SLUGS]
    result = _run(mutated, tmp_path)
    step = result.steps[0]
    assert not step.promoted
    # Rejected specifically because the gain did not generalize (or regressed).
    assert "OVERFIT" in step.reason or "regression" in step.reason


def test_generalizing_mutation_is_promoted(tmp_path):
    # Mutated: DERIVE now all pass AND HOLDOUT still all pass (holds).
    mutated = [_res(s, True) for s in _SLUGS]
    result = _run(mutated, tmp_path)
    step = result.steps[0]
    assert step.promoted
    assert "generalizes" in step.reason
