import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

from my_project_orchestrator.core.report import BuildReport
from my_project_orchestrator.core.assessment import (
    ProjectAssessment, HealthCheck, ProjectStructure, FeatureInventory,
    ProjectContext, TechnicalDebt, RiskAssessment,
)
from my_project_orchestrator.core.models import Task
from my_project_orchestrator.core.modes import BuildMode
from my_project_orchestrator.core.scratchpad import Scratchpad


def _make_assessment(**overrides):
    return ProjectAssessment(
        structure=overrides.get("structure", ProjectStructure(
            project_type="library", languages=["rust"]
        )),
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
    r.completed_tasks.append(Task(
        id="T-001", description="Add posting shard", project_ref="test",
        title="PostingShard impl", status="completed",
    ))
    r.failed_tasks.append(Task(
        id="T-002", description="Add query engine", project_ref="test",
        title="QueryEngine", status="failed",
    ))
    r.deferred_tasks.append(Task(
        id="T-003", description="Add benchmarks", project_ref="test",
        title="Benchmarks",
    ))
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
    r.health_after = HealthCheck(builds=True, test_count=10, test_failures=0, lint_warnings=2)
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
    r.key_decisions = ["Used parking_lot instead of std::sync", "Chose FxHashMap for perf"]
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


def test_report_debt_and_risk():
    r = _make_report()
    r.finalize(datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc))
    md = r.to_markdown()
    assert "Debt Score" in md
    assert "25/100" in md
    assert "Risk Level" in md
    assert "LOW" in md
