import tempfile
from pathlib import Path

from my_project_orchestrator.core.assessment import (
    HealthCheck,
    FeatureInfo,
    FeatureInventory,
    ProjectStructure,
    ProjectContext,
    TechnicalDebt,
    RiskAssessment,
    ProjectAssessment,
)
from my_project_orchestrator.core.models import Task, ExecutionResult
from my_project_orchestrator.analyzers.project_analyzer import _get_source_overview


def test_source_overview_covers_swift_and_csharp():
    # Previously the overview only saw py/js/ts/rs/go/java, so Swift/C# projects
    # looked empty. It must now surface them (and a structural symbol map).
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "Engine.swift").write_text("class Engine { func start() {} }\n")
        (td / "App.cs").write_text("public class App { public void Run() {} }\n")
        overview = _get_source_overview(td)
        assert "Engine.swift" in overview
        assert "App.cs" in overview
        # symbol map present when grammars are available
        if "Project structure" in overview:
            assert "class Engine" in overview and "class App" in overview


def test_source_overview_empty_project():
    with tempfile.TemporaryDirectory() as td:
        assert _get_source_overview(Path(td)) == "(no source files found)"


def test_health_check_defaults():
    h = HealthCheck()
    assert not h.builds
    assert not h.tests_pass
    assert h.test_count == 0
    assert h.lint_warnings == 0


def test_feature_info():
    f = FeatureInfo(
        name="auth", description="Authentication module", complexity="large"
    )
    assert f.name == "auth"
    assert f.complexity == "large"


def test_feature_inventory_empty():
    fi = FeatureInventory()
    assert fi.existing == []
    assert fi.incomplete == []
    assert fi.todos == []


def test_project_structure_defaults():
    s = ProjectStructure()
    assert s.project_type == "unknown"
    assert s.languages == []
    assert s.build_command is None


def test_project_context_defaults():
    c = ProjectContext()
    assert c.purpose == ""
    assert c.reference_impl is None


def test_technical_debt():
    td = TechnicalDebt(
        score=75, description="High debt", critical_issues=["god module"]
    )
    assert td.score == 75
    assert "god module" in td.critical_issues


def test_risk_assessment():
    r = RiskAssessment(level="high", factors=["no tests"], mitigations=["add tests"])
    assert r.level == "high"


def test_project_assessment_defaults():
    pa = ProjectAssessment()
    assert pa.structure.project_type == "unknown"
    assert pa.health.builds is False
    assert pa.tech_debt.score == 0
    assert pa.risk.level == "low"


def test_project_assessment_summary():
    pa = ProjectAssessment(
        structure=ProjectStructure(
            project_type="library", languages=["rust", "python"]
        ),
        health=HealthCheck(
            builds=True, test_count=50, test_failures=2, lint_warnings=3
        ),
    )
    s = pa.summary()
    assert "library" in s
    assert "rust" in s
    assert "build=OK" in s
    assert "48/50" in s
    assert "lint_warnings=3" in s


def test_project_assessment_summary_no_tests():
    pa = ProjectAssessment(
        structure=ProjectStructure(project_type="cli"),
        health=HealthCheck(builds=False),
    )
    s = pa.summary()
    assert "build=FAIL" in s
    assert "tests=none" in s


def test_task_model():
    t = Task(id="T-001", description="Implement posting shard", project_ref="hms")
    assert t.id == "T-001"
    assert t.status == "pending"
    assert t.dependencies == []
    assert t.complexity == "medium"
    assert t.category == "feature"


def test_task_model_with_fields():
    t = Task(
        id="T-002",
        description="Add query engine",
        project_ref="hms",
        title="QueryEngine",
        files_to_modify=["src/query.rs"],
        dependencies=["T-001"],
        complexity="large",
        category="core",
    )
    assert t.title == "QueryEngine"
    assert t.dependencies == ["T-001"]
    assert "src/query.rs" in t.files_to_modify


def test_execution_result():
    r = ExecutionResult(status="completed", message="All tests pass")
    assert r.status == "completed"
    assert r.start_time is not None
    assert r.end_time is None
