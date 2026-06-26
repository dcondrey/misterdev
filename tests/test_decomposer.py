from my_project_orchestrator.core.models import Task
from my_project_orchestrator.core.decomposer import (
    topological_sort,
    _add_implicit_dependencies,
    _parse_tasks,
    format_plan,
)
from my_project_orchestrator.core.modes import BuildMode


def _task(id, deps=None, category="feature", files_create=None, files_modify=None):
    return Task(
        id=id,
        description=id,
        project_ref=".",
        dependencies=deps or [],
        category=category,
        files_to_create=files_create or [],
        files_to_modify=files_modify or [],
    )


def test_topological_sort_linear():
    tasks = [
        _task("T-3", deps=["T-2"]),
        _task("T-1"),
        _task("T-2", deps=["T-1"]),
    ]
    result = topological_sort(tasks)
    assert [t.id for t in result] == ["T-1", "T-2", "T-3"]


def test_topological_sort_parallel():
    tasks = [
        _task("T-2", category="core"),
        _task("T-1", category="infrastructure"),
        _task("T-3", category="feature"),
    ]
    result = topological_sort(tasks)
    assert [t.id for t in result] == ["T-1", "T-2", "T-3"]


def test_topological_sort_cycle():
    tasks = [
        _task("A", deps=["B"]),
        _task("B", deps=["A"]),
    ]
    result = topological_sort(tasks)
    assert len(result) == 2


def test_add_implicit_dependencies():
    tasks = [
        _task("T-1", files_create=["src/new.py"]),
        _task("T-2", files_modify=["src/new.py"]),
    ]
    _add_implicit_dependencies(tasks)
    assert "T-1" in tasks[1].dependencies


def test_parse_tasks_json():
    response = '[{"id": "T-001", "title": "foo", "description": "bar"}]'
    tasks = _parse_tasks(response, ".")
    assert len(tasks) == 1
    assert tasks[0].id == "T-001"


def test_parse_tasks_prose_wrapped():
    response = 'Here are the tasks:\n[{"id": "T-001", "title": "foo"}]\nDone!'
    tasks = _parse_tasks(response, ".")
    assert len(tasks) == 1


def test_parse_tasks_fenced():
    response = '```json\n[{"id": "T-001", "title": "x"}]\n```'
    tasks = _parse_tasks(response, ".")
    assert len(tasks) == 1


def test_parse_tasks_invalid():
    assert _parse_tasks("not json at all", ".") == []


def test_format_plan():
    tasks = [_task("T-1", category="core")]
    plan = format_plan(tasks, BuildMode.COMPLETE)
    assert "T-1" in plan
    assert "complete" in plan.lower()


def test_ground_task_paths_reclassifies_existing_to_modify():
    import tempfile
    from pathlib import Path
    from my_project_orchestrator.core.decomposer import _ground_task_paths
    from my_project_orchestrator.core.models import Task

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
        (root / "src").mkdir()
        (root / "src" / "existing.rs").write_text("// real code\n", encoding="utf-8")
        t = Task(
            id="T-1",
            description="d",
            project_ref=td,
            files_to_create=["Cargo.toml", "src/existing.rs", "src/brand_new.rs"],
            files_to_modify=[],
        )
        _ground_task_paths([t], td)
        # Existing files reclassified to modify; genuinely new file stays create.
        assert "Cargo.toml" in t.files_to_modify
        assert "src/existing.rs" in t.files_to_modify
        assert "Cargo.toml" not in t.files_to_create
        assert "src/existing.rs" not in t.files_to_create
        assert t.files_to_create == ["src/brand_new.rs"]


def test_decompose_spec_honors_max_tasks_cap():
    import json
    from my_project_orchestrator.core.decomposer import decompose_spec
    from my_project_orchestrator.core.assessment import ProjectAssessment

    class _FakeClient:
        def generate_code(self, prompt, system_prompt=""):
            # Returns 10 tasks; the configured cap must trim to max_tasks.
            return json.dumps(
                [{"id": f"T-{i:03d}", "title": f"t{i}"} for i in range(1, 11)]
            )

    tasks = decompose_spec(
        "spec", ProjectAssessment(), BuildMode.DEBUG, _FakeClient(), ".", max_tasks=3
    )
    assert len(tasks) == 3


def test_decompose_includes_file_map_and_grounding_rule():
    # Regression for the "created a new duplicate file" failure: the decomposer
    # must be shown the real file map and told to use real paths.
    from my_project_orchestrator.core.decomposer import decompose_spec
    from my_project_orchestrator.core.assessment import (
        ProjectAssessment,
        HealthCheck,
        ProjectStructure,
        TechnicalDebt,
        RiskAssessment,
    )

    captured = {}

    class _Client:
        def generate_code(self, prompt, system=""):
            captured["prompt"] = prompt
            return '[{"id": "T-001", "title": "x", "description": "x", '\
                   '"acceptance_criteria": "x", "files_to_create": [], '\
                   '"files_to_modify": ["lib/allowlist.js"], "context_files": [], '\
                   '"dependencies": [], "complexity": "small", "category": "fix"}]'

    assessment = ProjectAssessment(
        structure=ProjectStructure(project_type="web-app", languages=["javascript"]),
        health=HealthCheck(builds=True, tests_pass=False),
        tech_debt=TechnicalDebt(score=10),
        risk=RiskAssessment(level="low"),
    )
    file_map = "lib/allowlist.js\n  function parseAllowlistCsv\n"
    tasks = decompose_spec(
        "fix parseAllowlistCsv", assessment, BuildMode.DEBUG, _Client(), ".",
        file_map=file_map,
    )
    assert "lib/allowlist.js" in captured["prompt"]
    assert "REAL paths" in captured["prompt"]
    assert tasks and tasks[0].files_to_modify == ["lib/allowlist.js"]


def test_decompose_without_file_map_uses_fallback_text():
    from my_project_orchestrator.core.decomposer import decompose_spec
    from my_project_orchestrator.core.assessment import (
        ProjectAssessment,
        HealthCheck,
        ProjectStructure,
        TechnicalDebt,
        RiskAssessment,
    )

    captured = {}

    class _Client:
        def generate_code(self, prompt, system=""):
            captured["prompt"] = prompt
            return "[]"

    assessment = ProjectAssessment(
        structure=ProjectStructure(project_type="library", languages=["python"]),
        health=HealthCheck(builds=True),
        tech_debt=TechnicalDebt(score=5),
        risk=RiskAssessment(level="low"),
    )
    decompose_spec("do x", assessment, BuildMode.SMART, _Client(), ".")
    assert "file map unavailable" in captured["prompt"]


def test_targets_prompt_helper():
    from my_project_orchestrator.core.decomposer import _targets_prompt
    section, rule = _targets_prompt(None)
    assert section == "" and rule == ""
    section, rule = _targets_prompt(
        [
            {"name": "core", "path": "emathy-core", "build_command": "cargo build"},
            {"name": "web", "path": "clients/web", "build_command": "npm run typecheck"},
        ]
    )
    assert "emathy-core/" in section and "clients/web/" in section
    assert "cargo build" in section and "npm run typecheck" in section
    assert "ONE target" in rule


def test_decompose_includes_targets_when_present():
    from my_project_orchestrator.core.decomposer import decompose_spec
    from my_project_orchestrator.core.assessment import (
        ProjectAssessment,
        HealthCheck,
        ProjectStructure,
        TechnicalDebt,
        RiskAssessment,
    )

    captured = {}

    class _Client:
        def generate_code(self, prompt, system=""):
            captured["prompt"] = prompt
            return "[]"

    assessment = ProjectAssessment(
        structure=ProjectStructure(project_type="monorepo", languages=["rust"]),
        health=HealthCheck(builds=True),
        tech_debt=TechnicalDebt(score=5),
        risk=RiskAssessment(level="low"),
    )
    decompose_spec(
        "x", assessment, BuildMode.SMART, _Client(), ".",
        targets=[{"name": "web", "path": "clients/web", "build_command": "npm run typecheck"}],
    )
    assert "clients/web/" in captured["prompt"]
    assert "ONE target" in captured["prompt"]
