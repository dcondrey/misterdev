"""Robust external task-list parsing across formats: JSON/YAML/Markdown/text,
ordered/unordered, one-line/multi-line, phases, field aliases, dependency
tables, dependency resolution, and the LLM fallback."""

from misterdev.core.planning.tasklist_parser import (
    _clean_path,
    _extract_dependency_table,
    detect_format,
    parse_task_list,
)

PR = "/proj"


def _p(text, name="tasks.md", llm=None):
    return parse_task_list(text, name, PR, llm=llm)


def test_clean_path_strips_backticks_annotations_and_prose_punctuation():
    # A backtick-quoted path followed by sentence punctuation must not strand an
    # inner backtick (the real-DEVPLAN bug: "`stats.ts`." -> "stats.ts`.").
    assert _clean_path("`stats.ts`.") == "stats.ts"
    assert _clean_path("`a.ts` (new)") == "a.ts"
    assert _clean_path("`x.ts`;") == "x.ts"
    assert _clean_path("src/f.ts).") == "src/f.ts"
    assert _clean_path("`p/q.ts`") == "p/q.ts"
    assert _clean_path("plain.ts") == "plain.ts"


def test_files_attr_cleans_backtick_quoted_prose_list():
    md = (
        "## T1: Types\n"
        "- **Files:** `packages/shared/src/events.ts`, `packages/shared/src/stats.ts`.\n"
    )
    t = _p(md)[0]
    assert t.files_to_modify == [
        "packages/shared/src/events.ts",
        "packages/shared/src/stats.ts",
    ]


# --- structured: JSON -------------------------------------------------------
def test_json_array():
    tasks = _p(
        '[{"id":"A","title":"Do A"},{"title":"Do B","depends_on":["A"]}]', "t.json"
    )
    assert [t.id for t in tasks] == ["A", "T-002"]
    assert tasks[1].dependencies == ["A"]  # resolved


def test_json_tasks_wrapper_and_aliases():
    js = (
        '{"tasks":[{"name":"Auth","success_criteria":"pytest passes",'
        '"relevant_files":["a.py","b.py"],"blocked_by":[]}]}'
    )
    t = _p(js, "t.json")[0]
    assert t.title == "Auth"
    assert t.acceptance_criteria == "pytest passes"  # success_criteria alias
    assert t.files_to_modify == ["a.py", "b.py"]  # relevant_files alias


def test_json_phased():
    js = (
        '{"phases":[{"name":"P1","tasks":[{"title":"one"}]},'
        '{"name":"P2","tasks":[{"title":"two","requires":["one"]}]}]}'
    )
    tasks = _p(js, "t.json")
    assert [t.title for t in tasks] == ["one", "two"]
    assert tasks[0].processor_data.get("phase") == "P1"
    assert tasks[1].dependencies == [tasks[0].id]  # "one" resolved by title


# --- structured: YAML -------------------------------------------------------
def test_yaml_list_with_deps():
    y = (
        "- id: setup\n  title: Set up\n  files: [x.py]\n"
        "- id: build\n  title: Build\n  depends_on: [setup]\n"
    )
    tasks = _p(y, "t.yaml")
    assert [t.id for t in tasks] == ["setup", "build"]
    assert tasks[1].dependencies == ["setup"]
    assert tasks[0].files_to_modify == ["x.py"]


# --- markdown ---------------------------------------------------------------
def test_markdown_ordered_list():
    md = "1. First task\n2. Second task\n3. Third task\n"
    tasks = _p(md)
    assert [t.title for t in tasks] == ["First task", "Second task", "Third task"]


def test_markdown_unordered_and_checkbox():
    md = "- [ ] Alpha\n- [x] Beta\n* Gamma\n"
    assert [t.title for t in _p(md)] == ["Alpha", "Beta", "Gamma"]


def test_markdown_headings_per_task_multiline_attrs():
    md = (
        "## Task 1: Implement login\n"
        "Build the login form.\n"
        "- files: src/login.py, src/forms.py\n"
        "- success criteria: pytest tests/test_login.py passes\n"
        "## Task 2: Wire it up\n"
        "- depends on: Task 1\n"
        "- files to create: src/router.py\n"
    )
    tasks = _p(md)
    assert tasks[0].title == "Implement login"
    assert "Build the login form." in tasks[0].description
    assert tasks[0].files_to_modify == ["src/login.py", "src/forms.py"]
    assert tasks[0].acceptance_criteria == "pytest tests/test_login.py passes"
    assert tasks[1].files_to_create == ["src/router.py"]
    assert tasks[1].dependencies == [tasks[0].id]  # "Task 1" -> ordinal 1


def test_markdown_phases():
    md = (
        "# Phase 1: Foundation\n"
        "- Set up the database\n"
        "- Create the schema\n"
        "# Phase 2: Features\n"
        "- Add the API\n"
    )
    tasks = _p(md)
    assert [t.title for t in tasks] == [
        "Set up the database",
        "Create the schema",
        "Add the API",
    ]
    assert tasks[0].processor_data["phase"] == "Phase 1: Foundation"
    assert tasks[2].processor_data["phase"] == "Phase 2: Features"


_DEVPLAN = """\
# Countless — v1 Development Plan

## Global Conventions

### Canonical constants (never guess)
- `RAW_RETENTION_DAYS = 90`
- Package `@countless/server` is the Worker.

### Secrets (must not guess)
- `ADMIN_TOKEN` is a secret, not a var.

## Wave 0 — Foundations

### T001 — Shared event and error types ‖ parallel
- **Description:** Define the shared types.
- **Files:** `packages/shared/src/types.ts` (new), `packages/shared/src/index.ts`
- **Completion:** `pnpm --filter @countless/shared typecheck` = 0

### T002 — Drizzle schema → D1 migration
- **Description:** Author the schema and generate the migration.
- **Files:** `apps/server/src/db/schema.ts`
- **Completion:** `pnpm --filter @countless/server db:generate` writes a migration

## Wave 1 — Ingest

### T003 — POST /api/collect handler (depends T001, T002)
- **Description:** Wire the ingest endpoint.
- **Completion:** `pnpm --filter @countless/server test` passes

### T062a — Analytics Engine binding
- **Completion:** binding present

### T062b — AE query lib ‖ parallel (after T062a)
- **Completion:** query returns rows

## Dependency Table

| Task | Title | Blocked by |
| --- | --- | --- |
| T001 | types | — |
| T002 | schema | — |
| T003 | collect | T001, T002 |
"""


def test_devplan_format_parses_as_written():
    tasks = _p(_DEVPLAN, name="DEVPLAN.md")
    by_id = {t.id: t for t in tasks}
    # Exactly the five real tasks — the global preamble and the trailing
    # dependency-table section did NOT become phantom tasks.
    assert sorted(by_id) == ["T001", "T002", "T003", "T062a", "T062b"]
    # Em-dash-id headings captured the id and stripped the ‖-parallel marker.
    assert by_id["T001"].title == "Shared event and error types"
    # `**Completion:**` mapped to the acceptance gate; backtick-annotated `**Files:**`
    # cleaned (the "(new)" note dropped).
    assert "typecheck" in by_id["T001"].acceptance_criteria
    assert by_id["T001"].files_to_create == ["packages/shared/src/types.ts"]
    assert "packages/shared/src/index.ts" in by_id["T001"].files_to_modify
    # Deps resolve from BOTH the heading "(depends ...)" and the table, and a
    # sub-task id (T062b) resolves its "(after T062a)".
    assert by_id["T003"].dependencies == ["T001", "T002"]
    assert by_id["T062b"].dependencies == ["T062a"]
    # Waves became phases; the preamble rode along as shared context.
    assert by_id["T001"].processor_data["phase"] == "Wave 0 — Foundations"
    shared = by_id["T003"].processor_data.get("shared_context", "")
    assert "RAW_RETENTION_DAYS = 90" in shared and "ADMIN_TOKEN" in shared


def test_markdown_dependency_table_unlocks_graph():
    md = (
        "## Build core\n## Add API\n## Write docs\n\n"
        "| Task | Blocked By |\n|------|-----------|\n"
        "| Add API | Build core |\n"
        "| Write docs | Add API |\n"
    )
    tasks = _p(md)
    ids = {t.title: t.id for t in tasks}
    api = next(t for t in tasks if t.title == "Add API")
    docs = next(t for t in tasks if t.title == "Write docs")
    assert api.dependencies == [ids["Build core"]]
    assert docs.dependencies == [ids["Add API"]]
    core = next(t for t in tasks if t.title == "Build core")
    assert core.dependencies == []  # independent -> parallelizable


def test_dependency_table_dep_column_first_no_self_edge():
    # Dependency column at index 0 with an unrecognized task header ('Step').
    # task_col must not blindly fall back to 0 (== dep_col) and key a self-edge.
    md = "| Blocked By | Step |\n|-----|-----|\n| T001 | T002 |\n"
    assert _extract_dependency_table(md) == {"T002": ["T001"]}


# --- plain text -------------------------------------------------------------
def test_plain_text_lines():
    txt = "Fix the parser bug\nAdd a regression test\nUpdate the changelog\n"
    assert [t.title for t in _p(txt, "t.txt")] == [
        "Fix the parser bug",
        "Add a regression test",
        "Update the changelog",
    ]


def test_plain_text_indented_attrs():
    t = _p("Setup\nImplement cache\n  files: cache.py", "t.txt")[1]
    assert t.files_to_modify == ["cache.py"]


# --- dependency resolution edge cases ---------------------------------------
def test_deps_resolve_by_id_title_and_number():
    js = (
        '[{"id":"first","title":"First"},'
        '{"id":"second","title":"Second","dependencies":["first"]},'
        '{"id":"third","title":"Third","dependencies":["Second","1"]}]'
    )
    tasks = _p(js, "t.json")
    assert tasks[1].dependencies == ["first"]
    assert set(tasks[2].dependencies) == {"second", "first"}  # title + ordinal


def test_unresolvable_dep_is_dropped_not_fatal():
    tasks = _p('[{"title":"X","dependencies":["ghost-task"]}]', "t.json")
    assert tasks[0].dependencies == []


# --- format detection + robustness ------------------------------------------
def test_detect_format():
    assert detect_format("x.json", "") == "json"
    assert detect_format("x.yaml", "") == "yaml"
    assert detect_format("x.md", "") == "markdown"
    assert detect_format("x.txt", "") == "text"
    assert detect_format("noext", '[{"a":1}]') == "json"  # content sniff
    assert detect_format("noext", "## Heading\n- item") == "markdown"


def test_malformed_json_falls_back_to_llm(monkeypatch):
    called = {}

    def fake_llm(prompt, system):
        called["yes"] = True
        return '[{"title":"Recovered task"}]'

    tasks = _p("{ this is not json", "t.json", llm=fake_llm)
    assert called.get("yes") and tasks[0].title == "Recovered task"


def test_empty_input_returns_empty_no_raise():
    assert _p("", "t.md") == []
    assert _p("   \n  \n", "t.txt") == []


# --- integration: TaskManager ingestion + topological ordering --------------
def test_taskmanager_load_and_topological_order(tmp_path):
    from types import SimpleNamespace
    from misterdev.core.task import TaskManager
    from misterdev.core.planning.decomposer import topological_sort

    f = tmp_path / "TASKS.md"
    f.write_text(
        "# Phase 1: Base\n"
        "## Build core\n- files: core.py\n"
        "## Add API\n- depends on: Build core\n- files: api.py\n"
        "## Write docs\n- depends on: Add API\n"
    )
    project = SimpleNamespace(
        path=tmp_path, config={"tasklist": "TASKS.md"}, llm_client=None
    )
    tm = TaskManager(project)
    tasks = tm.discover_tasks()  # dispatches to load_task_list via config
    assert len(tasks) == 3
    ordered = [t.title for t in topological_sort(tasks)]
    assert ordered.index("Build core") < ordered.index("Add API")
    assert ordered.index("Add API") < ordered.index("Write docs")
    # "Build core" is independent -> the engine can start it immediately/parallel
    core = next(t for t in tasks if t.title == "Build core")
    assert core.dependencies == []
    assert all(t.source_ref == str(f) for t in tasks)
