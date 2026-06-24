from my_project_orchestrator.agent import ProjectOrchestrator
from my_project_orchestrator.core.models import Task


class _FakeExec:
    """Drives suite-failure counts from a queue; reverts are recorded."""

    def __init__(self, fail_counts, unparseable=False):
        self._q = list(fail_counts)
        self._unparseable = unparseable
        self.reverted = []

    def _run_command(self, project, cmd, timeout=0):
        n = self._q.pop(0) if self._q else 0
        if self._unparseable:
            return False, "something went very wrong with no countable output"
        if n == 0:
            return True, "ℹ tests 10\nℹ pass 10\nℹ fail 0"
        return False, f"ℹ tests 10\nℹ pass {10 - n}\nℹ fail {n}"

    def find_task_commit(self, project, tid):
        return f"sha-{tid}"

    def revert_task_commit(self, project, sha):
        self.reverted.append(sha)
        return True


def _task(tid):
    return Task(id=tid, description="x", project_ref="p")


def test_suite_failures_green_red_unparseable():
    orch = ProjectOrchestrator()
    assert orch._suite_failures(None, _FakeExec([0]), "t", 1) == 0
    assert orch._suite_failures(None, _FakeExec([6]), "t", 1) == 6
    assert orch._suite_failures(None, _FakeExec([1], unparseable=True), "t", 1) is None


def test_count_gate_reverts_when_failures_rise():
    # Baseline 6; wave pushed it to 8 -> revert the wave task; recheck restores 6.
    orch = ProjectOrchestrator()
    ex = _FakeExec([8, 6])
    reverted = orch._integration_gate_count(
        None, ex, "t", [_task("T-1")], 1, baseline_failures=6
    )
    assert reverted == ["T-1"]
    assert ex.reverted == ["sha-T-1"]


def test_count_gate_accepts_when_not_worsened():
    orch = ProjectOrchestrator()
    ex = _FakeExec([6])  # same as baseline
    assert orch._integration_gate_count(None, ex, "t", [_task("T-1")], 1, 6) == []
    assert ex.reverted == []


def test_count_gate_accepts_improvement():
    orch = ProjectOrchestrator()
    ex = _FakeExec([3])  # fewer failures than baseline -> keep
    assert orch._integration_gate_count(None, ex, "t", [_task("T-1")], 1, 6) == []
    assert ex.reverted == []


def test_count_gate_unparseable_does_not_revert():
    orch = ProjectOrchestrator()
    ex = _FakeExec([9], unparseable=True)
    assert orch._integration_gate_count(None, ex, "t", [_task("T-1")], 1, 6) == []
    assert ex.reverted == []


def test_count_gate_reverts_multiple_until_restored():
    # Two wave tasks; failures 9 -> revert newest -> 8 -> revert next -> 6 (<=baseline).
    orch = ProjectOrchestrator()
    ex = _FakeExec([9, 8, 6])
    reverted = orch._integration_gate_count(
        None, ex, "t", [_task("T-1"), _task("T-2")], 1, baseline_failures=6
    )
    # Newest first: T-2 then T-1.
    assert reverted == ["T-2", "T-1"]


def test_integration_gate_dispatches_to_count_mode():
    orch = ProjectOrchestrator()
    ex = _FakeExec([8, 6])
    reverted = orch._integration_gate(
        None, ex, "t", [_task("T-1")], 1, baseline_failures=6
    )
    assert reverted == ["T-1"]


def test_integration_gate_green_baseline_uses_binary_path():
    # baseline_failures=0 -> binary path; a green suite reverts nothing.
    orch = ProjectOrchestrator()
    ex = _FakeExec([0])
    assert orch._integration_gate(None, ex, "t", [_task("T-1")], 1) == []
