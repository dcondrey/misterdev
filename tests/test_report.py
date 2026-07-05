from datetime import datetime, timezone

from misterdev.core.reporting.report import BuildReport
from misterdev.core.planning.assessment import (
    ProjectAssessment,
    HealthCheck,
    ProjectStructure,
    TechnicalDebt,
    RiskAssessment,
)
from misterdev.core.models import Task
from misterdev.core.modes import BuildMode
from misterdev.core.context.scratchpad import Scratchpad
from misterdev.core.verification.validator import ValidationResult


def _make_assessment(**overrides):
    return ProjectAssessment(
        structure=overrides.get(
            "structure", ProjectStructure(project_type="library", languages=["rust"])
        ),
        health=overrides.get("health", HealthCheck(builds=True, tests_pass=True)),
        tech_debt=TechnicalDebt(score=25),
        risk=RiskAssessment(level="low"),
    )


def _make_report(mode=BuildMode.COMPLETE, name="test-proj", **kw):
    return BuildReport(
        mode=mode,
        project_name=name,
        assessment=kw.get("assessment", _make_assessment()),
        start_time=kw.get("start_time", datetime(2026, 1, 1, tzinfo=timezone.utc)),
    )


def test_report_init():
    r = _make_report()
    assert r.project_name == "test-proj"
    assert r.mode == BuildMode.COMPLETE
    assert r.completed_tasks == []
    assert r.failed_tasks == []
    assert r.deferred_tasks == []


def test_report_finalize_sets_end_time():
    r = _make_report()
    r.finalize()
    assert r.end_time is not None
    assert r.end_time >= r.start_time


def test_report_finalize_with_explicit_time():
    r = _make_report()
    end = datetime(2026, 1, 2, tzinfo=timezone.utc)
    r.finalize(end)
    assert r.end_time == end


def test_report_to_markdown_basic():
    r = _make_report()
    r.finalize(datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc))
    md = r.to_markdown()
    assert "## Build Report" in md
    assert "test-proj" in md
    assert "library" in md
    assert "rust" in md
    assert "complete" in md
    assert "0/0 tasks completed" in md


def test_report_to_markdown_with_tasks():
    r = _make_report()
    r.completed_tasks.append(
        Task(
            id="T-001",
            description="Add posting shard",
            project_ref="test",
            title="PostingShard impl",
            status="completed",
        )
    )
    r.failed_tasks.append(
        Task(
            id="T-002",
            description="Add query engine",
            project_ref="test",
            title="QueryEngine",
            status="failed",
        )
    )
    r.deferred_tasks.append(
        Task(
            id="T-003",
            description="Add benchmarks",
            project_ref="test",
            title="Benchmarks",
        )
    )
    r.finalize(datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc))
    md = r.to_markdown()
    assert "PostingShard impl" in md
    assert "QueryEngine" in md
    assert "Benchmarks" in md
    assert "| Completed | 1 |" in md
    assert "| Failed | 1 |" in md
    assert "| Deferred | 1 |" in md


def test_report_to_markdown_with_health():
    r = _make_report()
    r.health_before = HealthCheck(builds=False, test_count=10, test_failures=3)
    r.health_after = HealthCheck(
        builds=True, test_count=10, test_failures=0, lint_warnings=2
    )
    r.finalize(datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc))
    md = r.to_markdown()
    assert "Health Before -> After" in md
    assert "NO" in md
    assert "YES" in md


def test_report_to_markdown_with_llm_usage():
    r = _make_report()
    r.llm_calls = 15
    r.llm_tokens = 50000
    r.llm_cost = 1.2345
    r.finalize(datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc))
    md = r.to_markdown()
    assert "LLM Usage" in md
    assert "15 calls" in md
    assert "50,000 tokens" in md
    assert "$1.2345" in md


def test_report_to_markdown_with_key_decisions():
    r = _make_report()
    r.key_decisions = [
        "Used parking_lot instead of std::sync",
        "Chose FxHashMap for perf",
    ]
    r.finalize(datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc))
    md = r.to_markdown()
    assert "Key Decisions" in md
    assert "parking_lot" in md


def test_report_to_markdown_with_scratchpad():
    r = _make_report()
    sp = Scratchpad()
    sp.record("convention", "Use thiserror for error types", "T-001")
    r.scratchpad = sp
    r.finalize(datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc))
    md = r.to_markdown()
    assert "Scratchpad Discoveries" in md
    assert "thiserror" in md


def test_report_duration_calculation():
    r = _make_report(start_time=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))
    r.finalize(datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc))
    md = r.to_markdown()
    assert "30.0 minutes" in md


def test_report_verdict_ship():
    r = _make_report()
    r.completed_tasks.append(
        Task(
            id="T-001",
            description="Add posting shard",
            project_ref="test",
            title="PostingShard impl",
            status="completed",
        )
    )
    r.validation_passed = True
    v = ValidationResult()
    v.build_ok = True
    v.tests_ok = True
    v.lint_ok = True
    r.validation = v
    r.finalize(datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc))
    md = r.to_markdown()
    assert "## Verdict: SHIP" in md
    assert "### Evidence" in md
    assert "1 completed, 0 failed, 0 deferred" in md
    assert "**Blocking items:**" not in md


def test_report_verdict_needs_review():
    r = _make_report()
    r.completed_tasks.append(
        Task(
            id="T-001",
            description="Add posting shard",
            project_ref="test",
            title="PostingShard impl",
            status="completed",
        )
    )
    r.failed_tasks.append(
        Task(
            id="T-002",
            description="Add query engine",
            project_ref="test",
            title="QueryEngine",
            status="failed",
        )
    )
    r.validation_passed = True
    v = ValidationResult()
    v.build_ok = True
    v.tests_ok = True
    r.validation = v
    r.finalize(datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc))
    md = r.to_markdown()
    assert "## Verdict: NEEDS REVIEW" in md
    assert "**Blocking items:**" in md
    assert "Failed task T-002" in md


def test_report_verdict_failed():
    r = _make_report()
    r.completed_tasks.append(
        Task(
            id="T-001",
            description="Add posting shard",
            project_ref="test",
            title="PostingShard impl",
            status="completed",
        )
    )
    r.validation_passed = False
    v = ValidationResult()
    v.issues = ["build failed: missing dependency"]
    r.validation = v
    r.finalize(datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc))
    md = r.to_markdown()
    assert "## Verdict: FAILED" in md
    assert "**Blocking items:**" in md
    assert "missing dependency" in md


def test_report_verdict_failed_when_nothing_done():
    r = _make_report()
    r.finalize(datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc))
    md = r.to_markdown()
    assert "## Verdict: FAILED" in md


def test_report_debt_and_risk():
    r = _make_report()
    r.finalize(datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc))
    md = r.to_markdown()
    assert "Debt Score" in md
    assert "25/100" in md
    assert "Risk Level" in md
    assert "LOW" in md


def test_report_init_degraded_subsystems_empty():
    r = _make_report()
    assert r.degraded_subsystems == []


def test_report_to_markdown_omits_degraded_when_empty():
    r = _make_report()
    r.finalize(datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc))
    md = r.to_markdown()
    assert "Degraded subsystems" not in md


def test_report_to_markdown_renders_degraded_when_populated():
    r = _make_report()
    r.degraded_subsystems = [
        "AB-MCTS planning: boom",
        "Empirical probes: kaboom",
    ]
    r.finalize(datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc))
    md = r.to_markdown()
    assert "Degraded subsystems" in md
    assert "AB-MCTS planning: boom" in md
    assert "Empirical probes: kaboom" in md
    assert "WITHOUT: AB-MCTS planning, Empirical probes" in md


def test_report_to_dict_includes_degraded_subsystems():
    r = _make_report()
    r.degraded_subsystems = ["Session audit: nope"]
    d = r.to_dict()
    assert d["degraded_subsystems"] == ["Session audit: nope"]


def test_report_to_dict_degraded_empty_by_default():
    r = _make_report()
    assert r.to_dict()["degraded_subsystems"] == []


def test_failure_reason_surfaces_logs_first_line():
    from misterdev.core.reporting.report import _failure_reason
    from misterdev.core.models import Task, ExecutionResult

    t = Task(id="T-1", description="x", project_ref="p", status="failed")
    t.execution_history.append(
        ExecutionResult(
            status="failed",
            message="Task failed after 3 attempts + escalation.",
            logs="SYNTAX ERROR in src/a.ts:\nUnexpected token at line 4\nmore",
        )
    )
    reason = _failure_reason(t)
    assert reason == "SYNTAX ERROR in src/a.ts:"


def test_failure_reason_escapes_pipes_and_falls_back():
    from misterdev.core.reporting.report import _failure_reason
    from misterdev.core.models import Task, ExecutionResult

    t = Task(id="T-2", description="x", project_ref="p", status="failed")
    assert _failure_reason(t) == "failed"  # no history -> status fallback
    t.execution_history.append(
        ExecutionResult(status="failed", message="a | b", logs="")
    )
    assert "\\|" in _failure_reason(t)  # pipe escaped for the table


def test_failed_tasks_table_includes_reason(tmp_path):
    from misterdev.core.reporting.report import BuildReport
    from misterdev.core.models import Task, ExecutionResult
    from misterdev.core.planning.assessment import (
        ProjectAssessment, HealthCheck, ProjectStructure, TechnicalDebt, RiskAssessment,
    )
    from misterdev.core.modes import BuildMode
    from datetime import datetime, timezone

    a = ProjectAssessment(
        structure=ProjectStructure(project_type="web-app"),
        health=HealthCheck(builds=True),
        tech_debt=TechnicalDebt(score=1),
        risk=RiskAssessment(level="low"),
    )
    rep = BuildReport(BuildMode.SMART, "x", a, datetime(2026, 1, 1, tzinfo=timezone.utc))
    t = Task(id="T-9", description="d", project_ref="p", status="failed", title="Do thing")
    t.execution_history.append(
        ExecutionResult(status="failed", message="m", logs="Build failed: cannot find module")
    )
    rep.failed_tasks.append(t)
    md = rep.to_markdown() if hasattr(rep, "to_markdown") else rep.render()
    assert "Build failed: cannot find module" in md and "Reason" in md
