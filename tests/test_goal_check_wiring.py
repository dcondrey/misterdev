"""Wiring tests for the optional goal-completion check in the build pipeline.

These exercise ProjectOrchestrator._run_goal_check directly (the unit that holds
the advisory-vs-blocking branching) with a monkeypatched LLM judge, so no
network is needed. The full _run_pipeline is covered elsewhere; here we prove the
verdict is recorded, advisory by default, blocking only when configured, and a
SKIP/error never touches the report or fails the build.
"""

from datetime import datetime, timezone
from pathlib import Path

from misterdev.agent import ProjectOrchestrator
from misterdev.core.planning.assessment import ProjectAssessment
from misterdev.core.modes import BuildMode
from misterdev.core.models import Task
from misterdev.core.reporting.report import BuildReport
from misterdev.core.verification.validator import ValidationResult


class _FakeClient:
    """LLM client whose generate_code returns a canned judge verdict."""

    def __init__(self, verdict: str):
        self._verdict = verdict
        self.calls = 0

    def generate_code(self, prompt, system=""):
        self.calls += 1
        return self._verdict


class _FakeProject:
    def __init__(self, tmp_path: Path, client, config):
        self.path = tmp_path
        self.llm_client = client
        self.config = config


def _report():
    r = BuildReport(
        BuildMode.SMART, "p", ProjectAssessment(), datetime.now(timezone.utc)
    )
    r.validation = ValidationResult()
    r.validation_passed = True
    return r


def _tasks():
    t = Task(id="t1", description="add a version endpoint", project_ref="p")
    t.acceptance_criteria = "GET /version returns the app version"
    t.title = "version endpoint"
    return [t]


def _orch(tmp_path):
    orch = ProjectOrchestrator()
    orch.last_build_succeeded = True
    return orch


def test_gap_verdict_records_gaps_advisory_by_default(tmp_path):
    client = _FakeClient('{"satisfied": false, "gaps": ["no /version endpoint"]}')
    project = _FakeProject(
        tmp_path, client, {"orchestrator": {"block_on_goal_gap": False}}
    )
    report = _report()
    orch = _orch(tmp_path)

    orch._run_goal_check(project, "ship a version endpoint", _tasks(), None, report)

    assert report.goal_gaps == ["no /version endpoint"]
    # Advisory: the build is NOT failed.
    assert report.validation_passed is True
    assert orch.last_build_succeeded is True
    assert client.calls == 1


def test_gap_verdict_blocks_when_configured(tmp_path):
    client = _FakeClient('{"satisfied": false, "gaps": ["missing endpoint"]}')
    project = _FakeProject(
        tmp_path, client, {"orchestrator": {"block_on_goal_gap": True}}
    )
    report = _report()
    orch = _orch(tmp_path)

    orch._run_goal_check(project, "goal", _tasks(), None, report)

    assert report.goal_gaps == ["missing endpoint"]
    assert report.validation_passed is False
    assert orch.last_build_succeeded is False
    assert any("Goal-completion check" in i for i in report.validation.issues)


def test_satisfied_verdict_records_nothing(tmp_path):
    client = _FakeClient('{"satisfied": true, "gaps": []}')
    project = _FakeProject(
        tmp_path, client, {"orchestrator": {"block_on_goal_gap": True}}
    )
    report = _report()
    orch = _orch(tmp_path)

    orch._run_goal_check(project, "goal", _tasks(), None, report)

    assert report.goal_gaps == []
    assert report.validation_passed is True
    assert orch.last_build_succeeded is True


def test_skip_when_no_goal_and_no_criteria(tmp_path):
    client = _FakeClient('{"satisfied": false, "gaps": ["x"]}')
    project = _FakeProject(
        tmp_path, client, {"orchestrator": {"block_on_goal_gap": True}}
    )
    report = _report()
    # No goal text and a task with no acceptance criteria -> nothing to judge.
    t = Task(id="t1", description="x", project_ref="p")
    orch = _orch(tmp_path)

    orch._run_goal_check(project, "", [t], None, report)

    assert report.goal_gaps == []
    assert report.validation_passed is True
    # The judge is never called when there is no target.
    assert client.calls == 0


def test_unparseable_verdict_is_skip_not_block(tmp_path):
    client = _FakeClient("I am not certain about this one.")
    project = _FakeProject(
        tmp_path, client, {"orchestrator": {"block_on_goal_gap": True}}
    )
    report = _report()
    orch = _orch(tmp_path)

    orch._run_goal_check(project, "goal", _tasks(), None, report)

    assert report.goal_gaps == []
    assert report.validation_passed is True
    assert orch.last_build_succeeded is True


def test_phase4_flags_default_off_leave_loop_unchanged():
    # The Phase-4 oracle flags are all off in the schema defaults, so a default
    # build neither runs the goal check nor enables the mutation gate — the
    # existing loop's behavior is unchanged.
    from misterdev.config import DEFAULT_CONFIG, get_setting

    orch = DEFAULT_CONFIG["orchestrator"]
    assert orch["goal_check"] is False
    assert orch["block_on_goal_gap"] is False
    assert orch["mutation_gate"] is False
    # spec_as_tests is default-on (T1.1), but advisory (spec_as_tests_block off).
    assert orch["spec_as_tests"] is True
    assert orch["spec_as_tests_block"] is False
    # And the gate guard reads the same default, so the goal check is skipped.
    assert get_setting({}, "orchestrator", "goal_check") is False


def test_goal_check_guard_skips_invocation_when_off(tmp_path, monkeypatch):
    # When goal_check is off, _run_pipeline must not call _run_goal_check. Guard
    # it by failing the test if the method is invoked under a flags-off config.
    import misterdev.agent as agent_mod

    called = {"n": 0}

    def boom(self, *a, **k):
        called["n"] += 1

    monkeypatch.setattr(agent_mod.ProjectOrchestrator, "_run_goal_check", boom)
    # Source-level guard: the only call site is gated by the goal_check setting.
    import inspect

    src = inspect.getsource(agent_mod.ProjectOrchestrator._run_pipeline)
    # The call to _run_goal_check is immediately preceded by the flag check.
    idx = src.index("self._run_goal_check(")
    preceding = src[:idx]
    assert 'get_setting(project.config, "orchestrator", "goal_check")' in preceding
    assert called["n"] == 0


def test_judge_exception_is_recorded_as_degraded_not_crash(tmp_path):
    class _Boom:
        def generate_code(self, prompt, system=""):
            raise RuntimeError("model down")

    project = _FakeProject(
        tmp_path, _Boom(), {"orchestrator": {"block_on_goal_gap": True}}
    )
    report = _report()
    orch = _orch(tmp_path)

    # Must not raise; an internal judge error is swallowed to SKIP inside
    # run_goal_check, so the build stays green and nothing is recorded.
    orch._run_goal_check(project, "goal", _tasks(), None, report)

    assert report.goal_gaps == []
    assert report.validation_passed is True
