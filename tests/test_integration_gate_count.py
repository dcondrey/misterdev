from misterdev.agent import ProjectOrchestrator
from misterdev.core.models import Task


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


# --- Identity-mode integration gate ----------------------------------------


class _Proj:
    config = {"language": "python"}


def _pytest_body(names):
    if not names:
        return "1 passed"
    return "\n".join(f"FAILED tests/t.py::{n} - AssertionError: x" for n in names)


def _ids(names):
    return ProjectOrchestrator()._failing_ids_from_output(_pytest_body(names), _Proj())


class _IdExec:
    """Drives failing-test-ID SETS from a queue; reverts recorded."""

    def __init__(self, id_sets, unparseable=False):
        self._q = list(id_sets)
        self._unparseable = unparseable
        self.reverted = []

    def _run_command(self, project, cmd, timeout=0, cwd=None):
        names = self._q.pop(0) if self._q else []
        if not names:
            return True, "1 passed"
        if self._unparseable:
            return False, "totally unparseable failure blob"
        return False, _pytest_body(names)

    def find_task_commit(self, project, tid):
        return f"sha-{tid}"

    def revert_task_commit(self, project, sha):
        self.reverted.append(sha)
        return True


def test_failing_ids_from_output_parses_pytest():
    ids = _ids(["test_alpha", "test_beta"])
    assert ids and len(ids) == 2
    # A green / unparseable output yields None (caller falls back to count).
    assert ProjectOrchestrator()._failing_ids_from_output("1 passed", _Proj()) is None


def test_identity_gate_reverts_offsetting_fix_break():
    # The case COUNT mode misses: baseline {alpha}; the wave fixed alpha but broke
    # beta — net count unchanged (1 -> 1) yet beta is a real new regression.
    orch = ProjectOrchestrator()
    baseline = _ids(["test_alpha"])
    ex = _IdExec([["test_beta"], ["test_alpha"]])  # post-wave, then after-revert
    reverted = orch._integration_gate_ids(_Proj(), ex, "t", [_task("T-1")], 1, baseline)
    assert reverted == ["T-1"]
    assert ex.reverted == ["sha-T-1"]


def test_identity_gate_accepts_genuine_fix():
    orch = ProjectOrchestrator()
    baseline = _ids(["test_alpha", "test_beta"])
    ex = _IdExec([["test_alpha"]])  # fixed beta, introduced nothing new
    assert (
        orch._integration_gate_ids(_Proj(), ex, "t", [_task("T-1")], 1, baseline) == []
    )
    assert ex.reverted == []


def test_identity_gate_no_progress_does_not_revert():
    # A no-op "fix" that resolved nothing but broke nothing: not reverted (can't
    # tell it from a legitimate feature wave), but surfaced as no-progress.
    orch = ProjectOrchestrator()
    baseline = _ids(["test_alpha"])
    ex = _IdExec([["test_alpha"]])
    assert (
        orch._integration_gate_ids(_Proj(), ex, "t", [_task("T-1")], 1, baseline) == []
    )
    assert ex.reverted == []


def test_identity_gate_reverts_new_failure():
    orch = ProjectOrchestrator()
    baseline = _ids(["test_alpha"])
    ex = _IdExec([["test_alpha", "test_beta"], ["test_alpha"]])
    reverted = orch._integration_gate_ids(_Proj(), ex, "t", [_task("T-1")], 1, baseline)
    assert reverted == ["T-1"]


def test_identity_gate_unparseable_after_does_not_revert():
    orch = ProjectOrchestrator()
    baseline = _ids(["test_alpha"])
    ex = _IdExec([["x"]], unparseable=True)
    assert (
        orch._integration_gate_ids(_Proj(), ex, "t", [_task("T-1")], 1, baseline) == []
    )
    assert ex.reverted == []


def test_integration_gate_prefers_identity_when_ids_present():
    # With a parsed baseline id-set on the project, the dispatcher uses identity
    # mode (revert on a new failure) even though the count is unchanged.
    orch = ProjectOrchestrator()
    proj = _Proj()
    proj.baseline_test_failing_ids = _ids(["test_alpha"])
    ex = _IdExec([["test_beta"], ["test_alpha"]])
    reverted = orch._integration_gate(
        proj, ex, "t", [_task("T-1")], 1, baseline_failures=1
    )
    assert reverted == ["T-1"]


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
    assert r(0, 0) is False  # green now
    assert r(0, 5) is False  # green now (was red)
    assert r(None, None) is False  # no countable baseline
    assert r(5, None) is False  # no countable baseline
    assert r(None, 0) is True  # binary fail from a green baseline
    assert r(None, 3) is False  # binary fail, but baseline was red -> can't compare
    assert r(7, 5) is True  # count rose
    assert r(5, 5) is False  # not worse


def test_integration_gate_targets_reverts_regressed_target_only():
    orch = ProjectOrchestrator()
    targets = [
        {"name": "web", "path": "clients/web", "test_command": "npm test"},
        {"name": "core", "path": "rust", "test_command": "cargo test"},
    ]
    # web regressed 0 -> green-fail handled via counts here: baseline 6, after 8.
    ex = _FakeExec([8, 6])  # web after=8 (>6) -> revert web task -> recheck=6
    web_task = _wtask("T-web", ["clients/web/src/a.ts"])
    core_task = _wtask(
        "T-core", ["rust/src/x.rs"]
    )  # core not exercised (no counts queued for it)
    reverted = orch._integration_gate_targets(
        _P(),
        ex,
        targets,
        [web_task, core_task],
        1,
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
            "targets": [
                {
                    "name": "apple",
                    "path": "clients/apple",
                    "build_command": "swift build",
                }
            ],
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


def test_validate_targets_runs_web_gate_red_fails(monkeypatch):
    from pathlib import Path
    import misterdev.core.verification.web_verify as wv

    class _Web:
        status, reason, evidence = "red", "submit button missing", None

    monkeypatch.setattr(wv, "run_web_gate", lambda path, cfg: _Web())
    orch = ProjectOrchestrator()
    orch._validate_executor = _FakeExec([0, 0])  # build/test green

    class _Proj:
        path = Path("/tmp")
        config = {
            "targets": [
                {
                    "name": "web",
                    "path": "clients/web",
                    "build_command": "tsc",
                    "web": {"url": "http://localhost:3000"},
                }
            ],
            "build": {},
        }
        target_baselines = {"web": 0}
        llm_client = None

    results = orch._validate_targets(_Proj(), None)
    assert results[0]["ok"] is False and "web verify" in results[0]["detail"]


def test_validate_targets_web_gate_green_passes(monkeypatch):
    from pathlib import Path
    import misterdev.core.verification.web_verify as wv

    class _Web:
        status, reason, evidence = "green", None, "shot.png"

    monkeypatch.setattr(wv, "run_web_gate", lambda path, cfg: _Web())
    orch = ProjectOrchestrator()
    orch._validate_executor = _FakeExec([0, 0])

    class _Proj:
        path = Path("/tmp")
        config = {
            "targets": [
                {
                    "name": "web",
                    "path": "clients/web",
                    "build_command": "tsc",
                    "web": {"url": "x"},
                }
            ],
            "build": {},
        }
        target_baselines = {"web": 0}
        llm_client = None

    results = orch._validate_targets(_Proj(), None)
    assert results[0]["ok"] is True
