import types
from datetime import datetime, timezone

from my_project_orchestrator.agent import ProjectOrchestrator
from my_project_orchestrator.core.reporting.report import BuildReport
from my_project_orchestrator.core.planning.assessment import (
    ProjectAssessment,
    HealthCheck,
    ProjectStructure,
    TechnicalDebt,
    RiskAssessment,
)
from my_project_orchestrator.core.modes import BuildMode
from my_project_orchestrator.llm.client import BudgetExceededError


def _assessment():
    return ProjectAssessment(
        structure=ProjectStructure(project_type="web-app", languages=["javascript"]),
        health=HealthCheck(builds=True, tests_pass=True),
        tech_debt=TechnicalDebt(score=10),
        risk=RiskAssessment(level="low"),
    )


def _fake_project(path):
    usage = types.SimpleNamespace(
        call_count=3, total_tokens=1000, estimated_cost=2.11, cache_read_tokens=0
    )
    client = types.SimpleNamespace(cumulative_usage=usage, cost_by_task={})
    return types.SimpleNamespace(path=path, name="rideshare", llm_client=client)


def test_halt_on_budget_with_report_returns_partial_markdown(tmp_path):
    # Regression (found dogfooding rideshare): a budget ceiling hit mid-pipeline
    # must degrade to a partial report, not crash the CLI with a traceback.
    orch = ProjectOrchestrator()
    report = BuildReport(
        BuildMode.SMART,
        "rideshare",
        _assessment(),
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    out = orch._halt_on_budget(
        _fake_project(tmp_path),
        report,
        BudgetExceededError("Budget of $2.00 exceeded (spent $2.11)"),
    )
    assert isinstance(out, str) and out
    assert orch.last_build_succeeded is False
    assert any("Halted by budget" in d for d in report.key_decisions)
    assert report.llm_cost == 2.11
    # A partial report was persisted.
    assert list((tmp_path / ".orchestrator" / "reports").glob("report_*.json"))


def test_halt_on_budget_without_report_returns_message(tmp_path):
    # Budget exhausted during analysis, before the report exists.
    orch = ProjectOrchestrator()
    out = orch._halt_on_budget(
        _fake_project(tmp_path),
        None,
        BudgetExceededError("Budget of $2.00 exceeded"),
    )
    assert "Build halted" in out
    assert "--budget" in out
    assert orch.last_build_succeeded is False
