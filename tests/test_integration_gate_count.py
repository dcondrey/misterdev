from my_project_orchestrator.agent import ProjectOrchestrator
from my_project_orchestrator.core.models import Task


class _FakeExec:
    """Drives suite-failure counts from a queue; reverts are recorded."""

    def __init__(self, fail_counts, unparseable=False):
        self._q = list(fail_counts)
        self._unparseable = unparseable
        self.reverted = []

    def _run_command(self, project, cmd, timeout=0, cwd=None):
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


# ----------------------------------------------------------------
# Per-target integration gate (polyglot)
# ----------------------------------------------------------------


class _P:
    from pathlib import Path as _Path

    path = _Path("/tmp")


def _wtask(tid, files):
    t = Task(id=tid, description="x", project_ref="p")
    t.files_to_modify = files
    return t


def test_target_regressed_helper():
    r = ProjectOrchestrator._target_regressed
    assert r(0, 0) is False           # green now
    assert r(0, 5) is False           # green now (was red)
    assert r(None, None) is False     # no countable baseline
    assert r(5, None) is False        # no countable baseline
    assert r(None, 0) is True         # binary fail from a green baseline
    assert r(None, 3) is False        # binary fail, but baseline was red -> can't compare
    assert r(7, 5) is True            # count rose
    assert r(5, 5) is False           # not worse


def test_integration_gate_targets_reverts_regressed_target_only():
    orch = ProjectOrchestrator()
    targets = [
        {"name": "web", "path": "clients/web", "test_command": "npm test"},
        {"name": "core", "path": "rust", "test_command": "cargo test"},
    ]
    # web regressed 0 -> green-fail handled via counts here: baseline 6, after 8.
    ex = _FakeExec([8, 6])  # web after=8 (>6) -> revert web task -> recheck=6
    web_task = _wtask("T-web", ["clients/web/src/a.ts"])
    core_task = _wtask("T-core", ["rust/src/x.rs"])  # core not exercised (no counts queued for it)
    reverted = orch._integration_gate_targets(
        _P(), ex, targets, [web_task, core_task], 1,
        {"web": 6, "core": 0},
    )
    assert reverted == ["T-web"]
    assert ex.reverted == ["sha-T-web"]


def test_integration_gate_targets_no_revert_when_target_green():
    orch = ProjectOrchestrator()
    targets = [{"name": "web", "path": "clients/web", "test_command": "npm test"}]
    ex = _FakeExec([0])  # web green after wave
    web_task = _wtask("T-web", ["clients/web/src/a.ts"])
    reverted = orch._integration_gate_targets(
        _P(), ex, targets, [web_task], 1, {"web": 0}
    )
    assert reverted == []
    assert ex.reverted == []


def test_integration_gate_targets_binary_fail_from_green_reverts():
    orch = ProjectOrchestrator()
    targets = [{"name": "web", "path": "clients/web", "build_command": "tsc"}]
    # Unparseable failure (typecheck) from a green baseline -> regression -> revert.
    ex = _FakeExec([1], unparseable=True)
    web_task = _wtask("T-web", ["clients/web/src/a.ts"])
    reverted = orch._integration_gate_targets(
        _P(), ex, targets, [web_task], 1, {"web": 0}
    )
    assert reverted == ["T-web"]


def test_validate_targets_ignores_pre_broken_target():
    # A target broken at BASELINE (e.g. Apple's pre-existing errors) must NOT fail
    # a run that never fixed it — only a genuine regression fails.
    from pathlib import Path

    orch = ProjectOrchestrator()
    orch._validate_executor = _FakeExec([1], unparseable=True)  # still broken now

    class _Proj:
        path = Path("/tmp")
        config = {
            "targets": [{"name": "apple", "path": "clients/apple", "build_command": "swift build"}],
            "build": {},
        }
        target_baselines = {"apple": None}  # was unparseable-broken at baseline

    results = orch._validate_targets(_Proj(), None)
    assert results == [{"name": "apple", "ok": True, "detail": "ok"}]


def test_validate_targets_flags_regression():
    from pathlib import Path

    orch = ProjectOrchestrator()
    orch._validate_executor = _FakeExec([1], unparseable=True)  # binary fail now

    class _Proj:
        path = Path("/tmp")
        config = {
            "targets": [{"name": "web", "path": "clients/web", "build_command": "tsc"}],
            "build": {},
        }
        target_baselines = {"web": 0}  # was green

    results = orch._validate_targets(_Proj(), None)
    assert results[0]["name"] == "web" and results[0]["ok"] is False
