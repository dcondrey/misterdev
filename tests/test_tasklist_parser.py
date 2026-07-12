"""Robust external task-list parsing across formats: JSON/YAML/Markdown/text,
ordered/unordered, one-line/multi-line, phases, field aliases, dependency
tables, dependency resolution, and the LLM fallback."""

from misterdev.core.planning.tasklist_parser import detect_format, parse_task_list

PR = "/proj"


def _p(text, name="tasks.md", llm=None):
    return parse_task_list(text, name, PR, llm=llm)


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
