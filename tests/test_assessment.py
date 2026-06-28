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
from my_project_orchestrator.analyzers.project_analyzer import (
    _get_source_overview,
    _leading_doc,
)


def test_leading_doc_extracts_rust_module_doc():
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "backend.rs"
        f.write_text(
            "//! Pure-Rust backend, used on wasm32.\n"
            "//! Degrades to empty output — the historical missing-model contract.\n"
            "use std::io;\nfn embed() {}\n"
        )
        doc = _leading_doc(f)
        assert "Degrades to empty output" in doc
        assert "use std::io" not in doc  # stops at first real code


def test_leading_doc_extracts_python_docstring():
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "mod.py"
        f.write_text('"""Token cache.\n\nKeyed by content hash."""\nimport os\n')
        doc = _leading_doc(f)
        assert "Token cache." in doc
        assert "import os" not in doc


def test_leading_doc_grafts_deep_intent_sentence():
    # The "intentional empty" signal often sits past the summary cap; it must be
    # grafted on so truncation never hides it (the exact stub-misclassification
    # this guards against).
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "backend.rs"
        filler = (
            "//! " + ("Background context that fills the summary budget. " * 8) + "\n"
        )
        f.write_text(
            filler
            + "//! On wasm the backend degrades to empty output by design; nothing panics.\n"
            + "fn embed() {}\n"
        )
        doc = _leading_doc(f)
        assert "degrades to empty output by design" in doc


def test_leading_doc_collects_intent_after_blank_separator():
    # A blank line between leading comment lines must NOT terminate collection,
    # or intent stated after the blank (the degrade/parity sentence) is lost.
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "backend.rs"
        f.write_text(
            "// Tract inference backend.\n"
            "\n"
            "// Degrades to empty on wasm by design; nothing panics.\n"
            "fn embed() {}\n"
        )
        doc = _leading_doc(f)
        assert "Degrades to empty on wasm by design" in doc


def test_leading_doc_empty_when_code_first():
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "a.rs"
        f.write_text("fn main() {}\n")
        assert _leading_doc(f) == ""


def test_leading_doc_ignores_c_preprocessor():
    # `#include` is a directive, not a comment — must not be read as doc intent.
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "a.c"
        f.write_text("#include <stdio.h>\nint main(){return 0;}\n")
        assert _leading_doc(f) == ""


def test_source_overview_surfaces_deep_file_intent():
    # Regression: a deep file documented as degrading-to-empty must reach the
    # analyzer as INTENT (so it is not mislabeled a stub), even when its body
    # falls past the source-heads char budget.
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        deep = td / "core" / "src" / "inference"
        deep.mkdir(parents=True)
        (deep / "tract_backend.rs").write_text(
            "//! wasm backend: degrades to empty by design; no panic.\n"
            "fn embed() -> Vec<f64> { Vec::new() }\n"
        )
        overview = _get_source_overview(td)
        assert "File intents" in overview
        assert "degrades to empty by design" in overview


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


def test_source_overview_reuses_provided_outline(monkeypatch):
    # When the caller passes a pre-built outline (the TopographyEngine graph), the
    # overview must reuse it and NOT parse a second throwaway SymbolGraph.
    import my_project_orchestrator.core.topography as topo

    def boom(*a, **k):
        raise AssertionError("SymbolGraph built despite a provided outline")

    monkeypatch.setattr(topo, "SymbolGraph", boom)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "a.rs").write_text("//! intent line\nfn x() {}\n")
        overview = _get_source_overview(td, outline="PREBUILT_OUTLINE")
        assert "PREBUILT_OUTLINE" in overview


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
