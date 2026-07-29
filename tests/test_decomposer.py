from misterdev.core.models import Task
from misterdev.core.planning.decomposer import (
    topological_sort,
    _add_implicit_dependencies,
    _find_cycle,
    _parse_tasks,
    _split_oversized_tasks,
    format_plan,
)
from misterdev.core.modes import BuildMode


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


def test_topological_sort_tiebreak_applies_mid_traversal():
    # A single root A unblocks both a `test` and a `feature` task in the same
    # wave. Category order (feature=2 before test=4) must win regardless of the
    # input order in which the dependents were declared.
    tasks = [
        _task("A", category="infrastructure"),
        _task("T-test", deps=["A"], category="test"),
        _task("T-feat", deps=["A"], category="feature"),
    ]
    result = topological_sort(tasks)
    order = [t.id for t in result]
    assert order.index("T-feat") < order.index("T-test"), order


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
    from misterdev.core.planning.decomposer import _ground_task_paths
    from misterdev.core.models import Task

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
    from misterdev.core.planning.decomposer import decompose_spec
    from misterdev.core.planning.assessment import ProjectAssessment

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
    from misterdev.core.planning.decomposer import decompose_spec
    from misterdev.core.planning.assessment import (
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
            return (
                '[{"id": "T-001", "title": "x", "description": "x", '
                '"acceptance_criteria": "x", "files_to_create": [], '
                '"files_to_modify": ["lib/allowlist.js"], "context_files": [], '
                '"dependencies": [], "complexity": "small", "category": "fix"}]'
            )

    assessment = ProjectAssessment(
        structure=ProjectStructure(project_type="web-app", languages=["javascript"]),
        health=HealthCheck(builds=True, tests_pass=False),
        tech_debt=TechnicalDebt(score=10),
        risk=RiskAssessment(level="low"),
    )
    file_map = "lib/allowlist.js\n  function parseAllowlistCsv\n"
    tasks = decompose_spec(
        "fix parseAllowlistCsv",
        assessment,
        BuildMode.DEBUG,
        _Client(),
        ".",
        file_map=file_map,
    )
    assert "lib/allowlist.js" in captured["prompt"]
    assert "REAL paths" in captured["prompt"]
    assert tasks and tasks[0].files_to_modify == ["lib/allowlist.js"]


def test_decompose_without_file_map_uses_fallback_text():
    from misterdev.core.planning.decomposer import decompose_spec
    from misterdev.core.planning.assessment import (
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
    from misterdev.core.planning.decomposer import _targets_prompt

    section, rule = _targets_prompt(None)
    assert section == "" and rule == ""
    section, rule = _targets_prompt(
        [
            {"name": "core", "path": "emathy-core", "build_command": "cargo build"},
            {
                "name": "web",
                "path": "clients/web",
                "build_command": "npm run typecheck",
            },
        ]
    )
    assert "emathy-core/" in section and "clients/web/" in section
    assert "cargo build" in section and "npm run typecheck" in section
    assert "ONE target" in rule


def test_decompose_includes_targets_when_present():
    from misterdev.core.planning.decomposer import decompose_spec
    from misterdev.core.planning.assessment import (
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
        "x",
        assessment,
        BuildMode.SMART,
        _Client(),
        ".",
        targets=[
            {"name": "web", "path": "clients/web", "build_command": "npm run typecheck"}
        ],
    )
    assert "clients/web/" in captured["prompt"]
    assert "ONE target" in captured["prompt"]


def test_cap_prunes_dangling_dependency_on_dropped_task():
    from misterdev.core.models import Task
    from misterdev.core.planning.decomposer import _cap_tasks

    def _t(tid, deps=None):
        return Task(
            id=tid, description="x", project_ref=".", dependencies=list(deps or [])
        )

    # C is the 3rd task -> dropped at cap 2; A's dependency on it must be pruned.
    tasks = [_t("A", deps=["C"]), _t("B"), _t("C")]
    out = _cap_tasks(tasks, 2)
    assert [x.id for x in out] == ["A", "B"]
    assert out[0].dependencies == []


def test_cap_noop_under_limit_preserves_deps():
    from misterdev.core.models import Task
    from misterdev.core.planning.decomposer import _cap_tasks

    tasks = [
        Task(id="A", description="x", project_ref=".", dependencies=["B"]),
        Task(id="B", description="x", project_ref="."),
    ]
    assert _cap_tasks(tasks, 5) is tasks
    assert tasks[0].dependencies == ["B"]


def _oversized_task(tid, n_modify=0, n_create=0, deps=None):
    return Task(
        id=tid,
        description=tid,
        project_ref=".",
        dependencies=list(deps or []),
        files_to_modify=[f"src/file_{i}.py" for i in range(n_modify)],
        files_to_create=[f"src/new_{i}.py" for i in range(n_create)],
    )


def test_split_oversized_tasks_noop_under_limit():
    tasks = [_oversized_task("T-1", n_modify=5)]
    result = _split_oversized_tasks(tasks, max_files=10)
    assert result is tasks


def test_split_oversized_tasks_partitions_files():
    tasks = [_oversized_task("T-1", n_modify=25)]
    result = _split_oversized_tasks(tasks, max_files=10)
    # Original task replaced by segments
    assert all(t.id != "T-1" for t in result)
    assert all(t.id.startswith("T-1-seg") for t in result)
    # All original files covered exactly once across segments
    original_files = set(tasks[0].files_to_modify)
    covered = set()
    for t in result:
        covered.update(t.files_to_modify + t.files_to_create)
    assert covered == original_files
    # Each segment within its limit
    for t in result:
        assert len(t.files_to_modify) + len(t.files_to_create) <= 10


def test_split_oversized_tasks_chains_segments():
    tasks = [_oversized_task("T-1", n_modify=25)]
    result = _split_oversized_tasks(tasks, max_files=10)
    # First segment inherits original deps; each subsequent depends on the previous
    assert result[0].dependencies == []
    for i in range(1, len(result)):
        assert result[i].dependencies == [result[i - 1].id]


def test_split_oversized_tasks_rewires_dependents():
    tasks = [
        _oversized_task("T-1", n_modify=25),
        _oversized_task("T-2", deps=["T-1"]),
    ]
    result = _split_oversized_tasks(tasks, max_files=10)
    t2 = next(t for t in result if t.id == "T-2")
    # T-2 should now depend on the LAST segment of T-1
    last_seg = sorted(t.id for t in result if t.id.startswith("T-1-seg"))[-1]
    assert last_seg in t2.dependencies
    assert "T-1" not in t2.dependencies


def test_add_implicit_dependencies_create_create_conflict():
    # Two tasks both declare they CREATE the same file; the second must depend on the first.
    tasks = [
        _task("T-1", files_create=["shared.py"]),
        _task("T-2", files_create=["shared.py"]),
    ]
    _add_implicit_dependencies(tasks)
    assert "T-1" in tasks[1].dependencies


def test_add_implicit_dependencies_no_self_dep():
    # A task that both creates AND modifies a file must not add itself as a dep.
    tasks = [_task("T-1", files_create=["f.py"], files_modify=["f.py"])]
    _add_implicit_dependencies(tasks)
    assert "T-1" not in tasks[0].dependencies


def test_find_cycle_returns_edge_path():
    tasks = [
        _task("A", deps=["B"]),
        _task("B", deps=["C"]),
        _task("C", deps=["A"]),
    ]
    task_map = {t.id: t for t in tasks}
    path = _find_cycle(tasks, task_map)
    assert len(path) >= 2
    # Path must form a cycle: last element equals first
    assert path[0] == path[-1]
    # All intermediate hops are real edges
    for i in range(len(path) - 1):
        assert path[i + 1] in task_map[path[i]].dependencies


def test_find_cycle_returns_empty_for_dag():
    tasks = [
        _task("A", deps=["B"]),
        _task("B"),
    ]
    task_map = {t.id: t for t in tasks}
    assert _find_cycle(tasks, task_map) == []
