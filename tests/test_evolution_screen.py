from dataclasses import dataclass

from misterdev.core.evolution.loop import Mutation
from misterdev.core.evolution.screen import MicroEvaluator, ScreenVerdict


@dataclass
class _Res:
    name: str
    resolved: bool


def _mut(paths=("misterdev/x.py",)):
    return Mutation(target="rust", paths=list(paths), patch="p", note="prompt")


class _Sandbox:
    """Fake apply/gates/run_only over a dict of case outcomes AFTER the mutation."""

    def __init__(self, gates_ok, outcomes):
        self.gates_ok = gates_ok
        self.outcomes = outcomes  # {case_id: resolved_bool}
        self.applied = 0
        self.torn_down = 0
        self.ran_only = None

    def apply(self, mutation):
        self.applied += 1
        return self._teardown

    def _teardown(self):
        self.torn_down += 1

    def gates(self):
        return self.gates_ok

    def run_only(self, only):
        self.ran_only = list(only)
        return [_Res(cid, self.outcomes.get(cid, False)) for cid in only], 0.01


def _evaluator(sb, targets, guards):
    return MicroEvaluator(
        apply=sb.apply,
        gates=sb.gates,
        run_only=sb.run_only,
        target_ids=list(targets),
        guard_ids=list(guards),
    )


def test_accepts_when_target_fixed_and_guard_intact():
    sb = _Sandbox(gates_ok=True, outcomes={"t1": True, "g1": True, "g2": True})
    v = _evaluator(sb, ["t1"], ["g1", "g2"]).screen(_mut())
    assert v.accepted
    assert v.targeted_resolved == 1 and v.guard_regressions == 0
    assert sb.torn_down == 1  # worktree always cleaned up
    # Only the targeted + guard cases were run, deduplicated.
    assert set(sb.ran_only) == {"t1", "g1", "g2"}


def test_rejects_when_no_target_fixed():
    sb = _Sandbox(gates_ok=True, outcomes={"t1": False, "g1": True})
    v = _evaluator(sb, ["t1"], ["g1"]).screen(_mut())
    assert not v.accepted and v.targeted_resolved == 0
    assert sb.torn_down == 1


def test_rejects_on_guard_regression():
    sb = _Sandbox(gates_ok=True, outcomes={"t1": True, "g1": False})
    v = _evaluator(sb, ["t1"], ["g1"]).screen(_mut())
    assert not v.accepted and v.guard_regressions == 1


def test_rejects_and_skips_benchmark_on_gate_failure():
    sb = _Sandbox(gates_ok=False, outcomes={"t1": True})
    v = _evaluator(sb, ["t1"], []).screen(_mut())
    assert not v.accepted and "gates failed" in v.reason
    assert sb.ran_only is None  # never spent the benchmark
    assert sb.torn_down == 1


def test_absent_case_is_not_counted_as_regression():
    # A guard case the harness did not report (slug mismatch) must not be counted
    # as a regression — that would reject a good mutation on a bookkeeping gap.
    class _Partial(_Sandbox):
        def run_only(self, only):
            self.ran_only = list(only)
            # 'g2' is silently dropped from results.
            return [_Res("t1", True), _Res("g1", True)], 0.0

    sb = _Partial(gates_ok=True, outcomes={})
    v = _evaluator(sb, ["t1"], ["g1", "g2"]).screen(_mut())
    assert v.accepted and v.guard_regressions == 0


def test_guardrail_rejects_protected_path():
    sb = _Sandbox(gates_ok=True, outcomes={"t1": True})
    v = _evaluator(sb, ["t1"], []).screen(_mut(paths=["tests/test_thing.py"]))
    assert not v.accepted and "guardrail" in v.reason
    assert sb.applied == 0  # never applied a guardrail-violating patch


def test_no_targets_cannot_accept():
    sb = _Sandbox(gates_ok=True, outcomes={})
    v = _evaluator(sb, [], ["g1"]).screen(_mut())
    assert not v.accepted and "no targets" in v.reason
    assert sb.applied == 0


def test_teardown_runs_even_on_benchmark_error():
    class _Boom(_Sandbox):
        def run_only(self, only):
            raise RuntimeError("benchmark exploded")

    sb = _Boom(gates_ok=True, outcomes={})
    v = _evaluator(sb, ["t1"], []).screen(_mut())
    assert not v.accepted and "evaluation failed" in v.reason
    assert sb.torn_down == 1


def test_rank_key_orders_by_targets_then_guard():
    a = ScreenVerdict(True, 3, 2, 5, 0, 0.0, "")
    b = ScreenVerdict(True, 3, 1, 5, 0, 0.0, "")
    assert a.rank_key > b.rank_key  # more targets fixed ranks higher
