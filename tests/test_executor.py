import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from my_project_orchestrator.task_executors.markdown_plan_executor import (
    MarkdownPlanExecutor,
    _detect_language,
    _diagnose_tampering,
    _diagnose_py_tampering,
    _count_tautologies,
    _is_test_file,
    _test_metrics,
    _extract_acceptance_command,
    _LANG_MAP,
    _merge_ranges,
    _window_lines,
    _relevant_line_ranges,
)
from my_project_orchestrator.core.models import Task
from my_project_orchestrator.core.scratchpad import Scratchpad


def test_detect_language_rust():
    assert _detect_language("src/core/posting.rs") == "rust"


def test_detect_language_python():
    assert _detect_language("main.py") == "python"
    assert _detect_language("stubs.pyi") == "python"


def test_detect_language_typescript():
    assert _detect_language("app.ts") == "typescript"
    assert _detect_language("component.tsx") == "typescript"


def test_detect_language_javascript():
    assert _detect_language("script.js") == "javascript"
    assert _detect_language("component.jsx") == "javascript"


def test_detect_language_go():
    assert _detect_language("main.go") == "go"


def test_detect_language_unknown():
    assert _detect_language("data.csv") == "text"
    assert _detect_language("README.md") == "text"
    assert _detect_language("Makefile") == "text"


def test_lang_map_completeness():
    assert ".rs" in _LANG_MAP
    assert ".py" in _LANG_MAP
    assert ".ts" in _LANG_MAP
    assert ".go" in _LANG_MAP
    assert ".java" in _LANG_MAP
    assert ".c" in _LANG_MAP
    assert ".cpp" in _LANG_MAP


def test_executor_init_default_scratchpad():
    e = MarkdownPlanExecutor()
    assert e.scratchpad is not None
    assert len(e.scratchpad) == 0


def test_executor_init_custom_scratchpad():
    sp = Scratchpad()
    sp.record("test", "discovery", "T-001")
    e = MarkdownPlanExecutor(scratchpad=sp)
    assert len(e.scratchpad) == 1


def test_snapshot_and_revert():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "src").mkdir()
        (td / "src" / "main.rs").write_text("fn main() {}", encoding="utf-8")
        (td / "src" / "lib.rs").write_text("pub fn hello() {}", encoding="utf-8")

        class FakeProject:
            path = td
            env_manager = None

        e = MarkdownPlanExecutor()
        proj = FakeProject()
        snapshot = e._snapshot_files(proj, ["src/main.rs", "src/lib.rs", "src/new.rs"])
        assert snapshot["src/main.rs"] == "fn main() {}"
        assert snapshot["src/lib.rs"] == "pub fn hello() {}"
        assert snapshot["src/new.rs"] is None

        (td / "src" / "main.rs").write_text("MODIFIED", encoding="utf-8")
        (td / "src" / "new.rs").write_text("NEW FILE", encoding="utf-8")

        e._revert_files(proj, snapshot)
        assert (td / "src" / "main.rs").read_text() == "fn main() {}"
        assert not (td / "src" / "new.rs").exists()


def test_read_file_for_context_existing():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "lib.rs").write_text("pub fn x() {}\n", encoding="utf-8")
        e = MarkdownPlanExecutor()
        ctx = e._read_file_for_context(td / "lib.rs", "lib.rs", max_lines=500)
        assert "lib.rs" in ctx
        assert "pub fn x()" in ctx


def test_read_file_for_context_missing():
    e = MarkdownPlanExecutor()
    ctx = e._read_file_for_context(Path("/nonexistent/foo.rs"), "foo.rs", max_lines=500)
    assert "Does not exist" in ctx


def test_read_file_for_context_truncation():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        content = "\n".join(f"line {i}" for i in range(100))
        (td / "big.py").write_text(content, encoding="utf-8")
        e = MarkdownPlanExecutor()
        ctx = e._read_file_for_context(td / "big.py", "big.py", max_lines=10)
        assert "truncated" in ctx
        assert "100 lines total" in ctx


def test_get_processor_config():
    e = MarkdownPlanExecutor()

    class FakeProject:
        config = {
            "task_processors": [
                {"type": "markdown_planner", "settings": {"max_retries_per_task": 5}}
            ]
        }

    cfg = e._get_processor_config(FakeProject())
    assert cfg["max_retries_per_task"] == 5


def test_get_processor_config_missing():
    e = MarkdownPlanExecutor()

    class FakeProject:
        config = {}

    cfg = e._get_processor_config(FakeProject())
    assert cfg == {}


def test_is_git_repo():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        class FakeProject:
            path = td

        e = MarkdownPlanExecutor()
        assert not e._is_git_repo(FakeProject())

        (td / ".git").mkdir()
        assert e._is_git_repo(FakeProject())


# ----------------------------------------------------------------
# Certainty-gated no-test completion (deterministic, no extra LLM call)
# ----------------------------------------------------------------


class _FakeLLMClient:
    def __init__(self, response: str):
        self._response = response

    @contextmanager
    def track_task(self, task_id):
        yield

    def generate(self, prompt, system_prompt=""):
        from my_project_orchestrator.llm.client import LLMResponse

        return LLMResponse(content=self._response, finish_reason="stop")

    def generate_code(self, prompt, system_prompt=""):
        return self.generate(prompt, system_prompt).content


class _FakeTaskManager:
    def __init__(self):
        self.statuses = {}

    def update_task_status(self, task_id, new_status):
        self.statuses[task_id] = new_status


class _FakeToolManager:
    tools = {}


class _FakeTopography:
    class _Graph:
        pass

    graph = _Graph()

    def initialize(self):
        pass

    def get_context_for_task(
        self, description, target_files, max_symbols=30, ranker=None
    ):
        return ""

    def get_file_outline(self, file_path):
        return ""

    def get_file_symbols(self, file_path):
        return []

    def get_project_outline(self):
        return ""


class _FakeProject:
    def __init__(self, path, response, config_extra=None):
        self.path = path
        self.env_manager = None
        self.llm_client = _FakeLLMClient(response)
        self.task_manager = _FakeTaskManager()
        self.tool_manager = _FakeToolManager()
        self.topography = _FakeTopography()
        self.config = {
            "prompt_templates": {
                "system": "system",
                "task_completion_instruction": "do the task",
                "error_correction_instruction": "fix the error",
            },
            "orchestrator": {"max_task_attempts": 1},
        }
        if config_extra:
            self.config.update(config_extra)


def _make_task():
    return Task(
        id="T-cert",
        description="trivial change",
        project_ref="p",
        strategy="surgical",
    )


def _run_no_test_task(td, response, config_extra=None):
    # No test_command, no build_command, prose-only response (no code blocks)
    # so no edits are applied. Reaches the no-test completion branch.
    task = _make_task()
    proj = _FakeProject(td, response, config_extra)
    # Pin strategy to surgical so escalation is exhausted immediately; the
    # decision under test is whether a single attempt is accepted as completed.
    task.processor_data["strategy"] = "surgical"
    e = MarkdownPlanExecutor()
    return e.execute(task, proj, use_git_branch=False)


def test_no_test_high_certainty_completes():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        response = (
            "I have verified this is correct. Tests pass successfully. "
            "The solution is complete and implemented."
        )
        result = _run_no_test_task(td, response)
        assert result.status == "completed"


def test_no_test_low_certainty_not_completed():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        response = "Maybe this could work, not sure, possibly wrong. Unclear."
        result = _run_no_test_task(td, response)
        assert result.status != "completed"


def test_certainty_threshold_configurable():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # A mid-confidence prose response that clears the default 0.5 gate but
        # not a stricter 0.95 gate, proving the threshold is read from config.
        response = "This works. Implemented and done."
        assert _run_no_test_task(td, response).status == "completed"
        strict = _run_no_test_task(
            td,
            response,
            config_extra={
                "orchestrator": {"max_task_attempts": 1, "certainty_threshold": 0.95}
            },
        )
        assert strict.status != "completed"


# ----------------------------------------------------------------
# Test-tampering guard (deterministic, no LLM call)
# ----------------------------------------------------------------


def test_is_test_file_patterns():
    assert _is_test_file("tests/test_foo.py")
    assert _is_test_file("test_foo.py")
    assert _is_test_file("foo_test.py")
    assert _is_test_file("conftest.py")
    assert _is_test_file("src/foo.test.ts")
    assert _is_test_file("foo_test.go")
    assert _is_test_file("foo_test.rs")
    assert _is_test_file("tests/integration/mod.rs")
    assert _is_test_file("com/example/FooTest.java")
    assert not _is_test_file("src/main.py")
    assert not _is_test_file("README.md")
    assert not _is_test_file("src/foo.ts")


def test_test_metrics_counts():
    py = (
        "import pytest\n"
        "def test_a():\n    assert 1 == 1\n    assert 2 == 2\n"
        "@pytest.mark.skip\n"
        "def test_b():\n    assert True\n"
    )
    tests, asserts, skips = _test_metrics(py)
    assert tests == 2
    assert asserts == 3
    assert skips == 1


def test_diagnose_tampering_deleted_test():
    before = "def test_a():\n    assert 1\ndef test_b():\n    assert 2\n"
    after = "def test_a():\n    assert 1\n"
    assert _diagnose_tampering(before, after) is not None


def test_diagnose_tampering_weakened_assertions():
    before = "def test_a():\n    assert x == 1\n    assert y == 2\n"
    after = "def test_a():\n    assert x == 1\n"
    assert "assertion count dropped" in _diagnose_tampering(before, after)


def test_diagnose_tampering_added_skip():
    before = "def test_a():\n    assert 1\n"
    after = "import pytest\n@pytest.mark.skip\ndef test_a():\n    assert 1\n"
    assert "skip" in _diagnose_tampering(before, after)


def test_diagnose_tampering_allows_additions():
    before = "def test_a():\n    assert 1\n"
    after = "def test_a():\n    assert 1\n    assert 2\ndef test_b():\n    assert 3\n"
    assert _diagnose_tampering(before, after) is None


def _edit_response(file_path: str, content: str) -> str:
    return f"```python\n# {file_path}\n{content}```\n"


def _run_edit_task(td, file_path, content, test_command, config_extra=None):
    task = _make_task()
    task.processor_data["strategy"] = "surgical"
    task.processor_data["test_command"] = test_command
    task.files_to_modify = [file_path]
    proj = _FakeProject(td, _edit_response(file_path, content), config_extra)
    e = MarkdownPlanExecutor()
    return e.execute(task, proj, use_git_branch=False)


def test_surgical_edit_applies_to_large_file_end_to_end():
    # A SEARCH/REPLACE response must land surgically: only the anchored line
    # changes and every other line is preserved byte-for-byte. This is the
    # whole point of the surgical path — a large file is edited without the
    # model reprinting it (which truncates past the output-token limit).
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Valid Rust so the (now tree-sitter-based) syntax gate passes; the file
        # is large enough to exercise the surgical path.
        original = "\n".join(f"const LINE_{i}: i32 = {i};" for i in range(2000)) + "\n"
        (td / "big.rs").write_text(original)
        response = (
            "```rust:big.rs\n"
            "<<<<<<< SEARCH\n"
            "const LINE_1500: i32 = 1500;\n"
            "=======\n"
            "const LINE_1500: i32 = 9999;\n"
            ">>>>>>> REPLACE\n"
            "```\n"
        )
        task = _make_task()
        task.processor_data["strategy"] = "surgical"
        task.processor_data["test_command"] = "true"
        task.files_to_modify = ["big.rs"]
        proj = _FakeProject(td, response)
        result = MarkdownPlanExecutor().execute(task, proj, use_git_branch=False)

        assert result.status == "completed"
        new = (td / "big.rs").read_text()
        assert "const LINE_1500: i32 = 9999;" in new
        # every other line preserved: same line count, only one line differs
        assert new.count("\n") == original.count("\n")
        o_lines, n_lines = original.splitlines(), new.splitlines()
        diffs = [i for i, (a, b) in enumerate(zip(o_lines, n_lines)) if a != b]
        assert diffs == [1500]


def test_code_context_injects_file_outline():
    # The model editing a large file should receive a symbol outline (table of
    # contents) so it can navigate and anchor SEARCH blocks precisely.
    from types import SimpleNamespace
    from my_project_orchestrator.core.topography import TopographyEngine

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = "\n".join(["fn start() {}"] + ["x"] * 80 + ["fn stop() {}"])
        (td / "engine.rs").write_text(src)
        topo = TopographyEngine(td, llm_client=None)
        project = SimpleNamespace(path=td, config={}, topography=topo)
        ctx = MarkdownPlanExecutor()._get_code_context(project, ["engine.rs"], [])
        if "function start" not in ctx:
            return  # rust grammar unavailable in this environment
        assert "Outline of engine.rs" in ctx
        assert "function stop" in ctx
        # full file body is still present for exact anchoring
        assert ctx.count("fn start() {}") >= 1


def test_merge_ranges_merges_overlapping_and_adjacent():
    assert _merge_ranges([(0, 5), (3, 8), (20, 25)]) == [(0, 8), (20, 25)]
    assert _merge_ranges([(0, 5), (6, 9)]) == [(0, 9)]  # adjacent
    assert _merge_ranges([(10, 12), (0, 2)]) == [(0, 2), (10, 12)]  # sorted


def test_window_lines_keeps_spans_and_marks_elisions():
    lines = [f"L{i}" for i in range(100)]
    out = _window_lines(lines, [(0, 2), (50, 52)])
    assert "L0" in out and "L50" in out
    assert "L25" not in out
    assert "elided" in out  # gap between the two kept spans is marked


class _Sym:
    def __init__(self, name, start, end):
        self.name = name
        self.start_line = start
        self.end_line = end


def test_relevant_line_ranges_token_match_not_substring():
    syms = [_Sym("SceneLocator", 400, 410), _Sym("Loc", 800, 810)]

    class _T:
        description = "Refactor the SceneLocator handler"
        acceptance_criteria = ""

    ranges = _relevant_line_ranges(syms, _T(), 1000)
    # SceneLocator matches as a whole token; "Loc" must NOT (substring only)
    assert any(400 <= a <= 410 or a <= 400 <= b for a, b in ranges)
    assert not any(800 <= a for a, b in ranges)


def test_relevant_line_ranges_none_when_no_match():
    syms = [_Sym("Foo", 10, 20)]

    class _T:
        description = "unrelated work"
        acceptance_criteria = ""

    assert _relevant_line_ranges(syms, _T(), 100) is None


def test_render_large_file_windows_to_relevant_symbol():
    from types import SimpleNamespace
    from my_project_orchestrator.core.topography import TopographyEngine

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = (
            ["fn alpha() {}"]
            + ["// filler"] * 900
            + ["fn target_handler() {", "    let z = 1;", "}"]
            + ["// tail"] * 50
        )
        (td / "big.rs").write_text("\n".join(src))
        topo = TopographyEngine(td, llm_client=None)
        project = SimpleNamespace(
            path=td,
            config={"orchestrator": {"large_file_line_threshold": 800}},
            topography=topo,
        )
        task = _make_task()
        task.description = "Fix a bug in target_handler"
        ctx = MarkdownPlanExecutor()._get_code_context(
            project, ["big.rs"], [], task=task
        )
        if "function target_handler" not in ctx:
            return  # rust grammar unavailable
        assert "fn target_handler() {" in ctx  # relevant body shown verbatim
        assert "elided" in ctx  # filler elided
        assert len(ctx) < len("\n".join(src))  # smaller than the full file


def test_render_large_file_no_match_sends_full():
    from types import SimpleNamespace
    from my_project_orchestrator.core.topography import TopographyEngine

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = ["fn alpha() {}"] + ["// filler"] * 900
        (td / "big.rs").write_text("\n".join(src))
        topo = TopographyEngine(td, llm_client=None)
        project = SimpleNamespace(
            path=td,
            config={"orchestrator": {"large_file_line_threshold": 800}},
            topography=topo,
        )
        task = _make_task()
        task.description = "unrelated change with no symbol names"
        ctx = MarkdownPlanExecutor()._get_code_context(
            project, ["big.rs"], [], task=task
        )
        assert "elided" not in ctx  # fell back to the full file


def test_surgical_edit_conflict_does_not_write_partial():
    # If the SEARCH anchor does not match, the file on disk must be left
    # untouched (no partial/truncated write) and the task must not complete.
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        original = "fn main() {}\n"
        (td / "m.rs").write_text(original)
        response = (
            "```rust:m.rs\n"
            "<<<<<<< SEARCH\n"
            "this text is not in the file\n"
            "=======\n"
            "garbage\n"
            ">>>>>>> REPLACE\n"
            "```\n"
        )
        task = _make_task()
        task.processor_data["strategy"] = "surgical"
        task.processor_data["test_command"] = "true"
        task.files_to_modify = ["m.rs"]
        proj = _FakeProject(td, response)
        result = MarkdownPlanExecutor().execute(task, proj, use_git_branch=False)

        assert result.status != "completed"
        assert (td / "m.rs").read_text() == original


def test_tampering_edit_rejected_does_not_complete():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        original = (
            "def test_a():\n    assert 1 == 1\n\n\ndef test_b():\n    assert 2 == 2\n"
        )
        (td / "test_thing.py").write_text(original, encoding="utf-8")
        # The LLM deletes test_b but a `true` gate would "pass". Guard must fire
        # before the gate, so the task does not complete and the file is intact.
        gutted = "def test_a():\n    assert 1 == 1\n"
        result = _run_edit_task(td, "test_thing.py", gutted, "true")
        assert result.status != "completed"
        assert (td / "test_thing.py").read_text() == original


def test_weakened_assertion_edit_rejected():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        original = "def test_a():\n    assert x == 1\n    assert y == 2\n"
        (td / "test_thing.py").write_text(original, encoding="utf-8")
        weakened = "def test_a():\n    assert x == 1\n"
        result = _run_edit_task(td, "test_thing.py", weakened, "true")
        assert result.status != "completed"


def test_added_skip_marker_edit_rejected():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        original = "def test_a():\n    assert x == 1\n"
        (td / "test_thing.py").write_text(original, encoding="utf-8")
        skipped = "import pytest\n@pytest.mark.skip\ndef test_a():\n    assert x == 1\n"
        result = _run_edit_task(td, "test_thing.py", skipped, "true")
        assert result.status != "completed"


def test_adding_new_tests_allowed():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        original = "def test_a():\n    assert 1 == 1\n"
        (td / "test_thing.py").write_text(original, encoding="utf-8")
        grown = (
            "def test_a():\n    assert 1 == 1\n\n\ndef test_b():\n    assert 2 == 2\n"
        )
        result = _run_edit_task(td, "test_thing.py", grown, "true")
        assert result.status == "completed"
        assert "test_b" in (td / "test_thing.py").read_text()


# ----------------------------------------------------------------
# Per-task type-check gate (deterministic, no LLM call)
# ----------------------------------------------------------------


def _run_typecheck_task(td, file_path, content, typecheck_command, test_command=None):
    task = _make_task()
    task.processor_data["strategy"] = "surgical"
    task.processor_data["typecheck_command"] = typecheck_command
    if test_command is not None:
        task.processor_data["test_command"] = test_command
    task.files_to_modify = [file_path]
    proj = _FakeProject(td, _edit_response(file_path, content))
    e = MarkdownPlanExecutor()
    return e.execute(task, proj, use_git_branch=False)


def test_typecheck_failure_does_not_complete():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Type check fails, so the task must not complete even though a `true`
        # test gate would pass; the failure forces a retry instead.
        result = _run_typecheck_task(
            td, "mod.py", "x = 1\n", "false", test_command="true"
        )
        assert result.status != "completed"


def test_typecheck_pass_then_tests_complete():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        result = _run_typecheck_task(
            td, "mod.py", "x = 1\n", "true", test_command="true"
        )
        assert result.status == "completed"


def test_no_typecheck_command_unchanged():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # No typecheck command configured: behaves exactly as before, the test
        # gate alone decides completion.
        result = _run_edit_task(td, "mod.py", "x = 1\n", "true")
        assert result.status == "completed"


def test_allow_test_edits_escape_hatch():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        original = "def test_a():\n    assert 1\ndef test_b():\n    assert 2\n"
        (td / "test_thing.py").write_text(original, encoding="utf-8")
        gutted = "def test_a():\n    assert 1\n"
        result = _run_edit_task(
            td,
            "test_thing.py",
            gutted,
            "true",
            config_extra={
                "orchestrator": {"max_task_attempts": 1, "allow_test_edits": True}
            },
        )
        assert result.status == "completed"


# ----------------------------------------------------------------
# Per-task acceptance-criteria gate (deterministic command path)
# ----------------------------------------------------------------


def test_extract_acceptance_command_pytest():
    assert (
        _extract_acceptance_command("pytest tests/test_auth.py passes")
        == "pytest tests/test_auth.py"
    )


def test_extract_acceptance_command_with_lead_in_prose():
    assert (
        _extract_acceptance_command("Verify that cargo test --lib auth passes")
        == "cargo test --lib auth"
    )


def test_extract_acceptance_command_backticked():
    assert (
        _extract_acceptance_command("Run `python -m pytest tests/` to confirm")
        == "python -m pytest tests/"
    )


def test_extract_acceptance_command_free_text_none():
    assert _extract_acceptance_command("the login form rejects empty passwords") is None
    assert _extract_acceptance_command("") is None
    assert _extract_acceptance_command("users can reset their password") is None


def _run_acceptance_task(
    td, file_path, content, test_command, acceptance_criteria, config_extra=None
):
    task = _make_task()
    task.processor_data["strategy"] = "surgical"
    task.processor_data["test_command"] = test_command
    task.acceptance_criteria = acceptance_criteria
    task.files_to_modify = [file_path]
    proj = _FakeProject(td, _edit_response(file_path, content), config_extra)
    e = MarkdownPlanExecutor()
    return e.execute(task, proj, use_git_branch=False)


def test_acceptance_command_passes_completes():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        result = _run_acceptance_task(
            td,
            "mod.py",
            "x = 1\n",
            "true",
            f"{sys.executable} -m pytest --version passes",
        )
        assert result.status == "completed"


def test_acceptance_command_fails_does_not_complete():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Test gate passes but the acceptance command exits non-zero, so the
        # task must not complete on this attempt (single attempt -> failed).
        result = _run_acceptance_task(
            td, "mod.py", "x = 1\n", "true", "pytest /no/such/path.py passes"
        )
        assert result.status != "completed"


def test_acceptance_unparseable_criteria_completes_as_before():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Free-text criterion has no runnable command: must not block, behaves
        # exactly as before (test gate alone decides completion).
        result = _run_acceptance_task(
            td, "mod.py", "x = 1\n", "true", "the module exposes a clean API"
        )
        assert result.status == "completed"


def test_acceptance_empty_criteria_completes_as_before():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        result = _run_acceptance_task(td, "mod.py", "x = 1\n", "true", "")
        assert result.status == "completed"


def test_acceptance_gate_disabled_completes():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # A failing acceptance command is ignored when the gate is turned off.
        result = _run_acceptance_task(
            td,
            "mod.py",
            "x = 1\n",
            "true",
            "pytest /no/such/path.py passes",
            config_extra={
                "orchestrator": {"max_task_attempts": 1, "verify_acceptance": False}
            },
        )
        assert result.status == "completed"


def test_acceptance_gate_no_test_branch():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # No test_command at all: acceptance runs on the no-test completion
        # branch too. A failing acceptance command blocks completion.
        task = _make_task()
        task.processor_data["strategy"] = "surgical"
        task.acceptance_criteria = "pytest /no/such/path.py passes"
        response = "I have verified this is correct. Implemented and complete."
        proj = _FakeProject(td, response)
        e = MarkdownPlanExecutor()
        result = e.execute(task, proj, use_git_branch=False)
        assert result.status != "completed"


# ---------------------------------------------------------------------------
# Strengthened tamper guard: per-function AST accounting + tautology detection
# ---------------------------------------------------------------------------


def test_py_tampering_detects_removed_test_masked_by_addition():
    # test_b is deleted but a new test_c is added, so the file's TOTAL test and
    # assertion counts do not drop -- the per-function AST check must still flag
    # the removed existing test (regex totals alone would miss this).
    before = "def test_a():\n    assert x == 1\n\ndef test_b():\n    assert y == 2\n"
    after = "def test_a():\n    assert x == 1\n\ndef test_c():\n    assert z == 3\n"
    assert _diagnose_tampering(before, after) is None  # totals unchanged
    reason = _diagnose_py_tampering(before, after)
    assert reason and "assertion" in reason  # `assert y == 2` vanished


def test_py_tampering_detects_weakened_test_masked_by_addition():
    before = "def test_a():\n    assert x == 1\n    assert x > 0\n"
    after = (
        "def test_a():\n    assert x == 1\n\n"
        "def test_new():\n    assert q\n    assert r\n"
    )
    reason = _diagnose_py_tampering(before, after)
    assert reason and "assertion" in reason  # `assert x > 0` vanished


def test_py_tampering_allows_rename():
    # Renaming a test keeps its assertions, so it must not be flagged.
    before = "def test_old():\n    assert f() == 1\n"
    after = "def test_new():\n    assert f() == 1\n"
    assert _diagnose_py_tampering(before, after) is None


def test_py_tampering_allows_split_and_merge():
    # Splitting one test into two (or merging) preserves the assertions.
    before = "def test_big():\n    assert a()\n    assert b()\n"
    after = "def test_a():\n    assert a()\n\ndef test_b():\n    assert b()\n"
    assert _diagnose_py_tampering(before, after) is None
    assert _diagnose_py_tampering(after, before) is None  # merge direction too


def test_py_tampering_allows_parametrization():
    before = (
        "def test_sq_1():\n    assert sq(1) == 1\n"
        "def test_sq_2():\n    assert sq(2) == 4\n"
    )
    after = (
        "import pytest\n"
        "@pytest.mark.parametrize('n,e', [(1, 1), (2, 4)])\n"
        "def test_sq(n, e):\n    assert sq(n) == e\n"
    )
    assert _diagnose_py_tampering(before, after) is None


def test_py_tampering_flags_changed_expected_value():
    # Editing a test's expected value to match buggy output is the gaming
    # vector and must still be caught.
    before = "def test_a():\n    assert compute() == 5\n"
    after = "def test_a():\n    assert compute() == 4\n"
    assert _diagnose_py_tampering(before, after)


def test_py_tampering_allows_pure_addition():
    before = "def test_a():\n    assert x == 1\n"
    after = "def test_a():\n    assert x == 1\n\ndef test_b():\n    assert y == 2\n"
    assert _diagnose_py_tampering(before, after) is None


def test_py_tampering_none_on_unparseable():
    assert _diagnose_py_tampering("def test_a(:\n  bad", "def test_a(): pass") is None


def test_tautology_counter_and_diagnosis():
    assert _count_tautologies("assert True\n") == 1
    assert _count_tautologies("    assert x == y\n") == 0
    before = "def test_a():\n    assert compute() == 5\n"
    after = "def test_a():\n    assert True\n"
    # assertion count holds but a real check became trivially true.
    assert _diagnose_tampering(before, after)


def test_inbody_skip_counts_as_skip_growth():
    before = "def test_a():\n    assert x == 1\n"
    after = "def test_a():\n    import pytest\n    pytest.skip('later')\n    assert x == 1\n"
    assert _diagnose_tampering(before, after)


def test_judge_affordable_respects_budget_floor():
    e = MarkdownPlanExecutor()

    class _Client:
        pass

    class _Proj:
        pass

    proj = _Proj()
    client = _Client()
    proj.llm_client = client
    client._budget = 100.0
    client.budget_remaining = 50.0
    assert e._judge_affordable(proj) is True
    client.budget_remaining = 5.0  # 5% < 10% floor
    assert e._judge_affordable(proj) is False
    client.budget_remaining = None  # unknown -> fail open
    assert e._judge_affordable(proj) is True


# ---------------------------------------------------------------------------
# Golden suite: model-blind, immutable test protection
# ---------------------------------------------------------------------------

from my_project_orchestrator.task_executors.markdown_plan_executor import (  # noqa: E402
    _is_golden_path,
)


def test_is_golden_path_matching():
    pats = ["tests/golden/", "tests/test_contract_*.py"]
    assert _is_golden_path("tests/golden/test_x.py", pats)
    assert _is_golden_path("tests/golden/sub/test_y.py", pats)
    assert _is_golden_path("tests/test_contract_auth.py", pats)
    assert not _is_golden_path("tests/test_other.py", pats)
    assert not _is_golden_path("src/main.py", pats)
    assert not _is_golden_path("anything.py", [])


def test_validate_edit_paths_rejects_golden():
    with tempfile.TemporaryDirectory() as td:

        class _P:
            path = Path(td)
            config = {"orchestrator": {"golden_paths": ["tests/golden/"]}}

        task = Task(
            id="T",
            description="d",
            project_ref="p",
            files_to_modify=["src/a.py"],
        )
        e = MarkdownPlanExecutor()
        edits = {
            "src/a.py": "x = 1\n",
            "tests/golden/test_contract.py": "assert False\n",
        }
        valid = e._validate_edit_paths(_P(), task, edits)
        assert "src/a.py" in valid
        assert "tests/golden/test_contract.py" not in valid


def test_golden_context_file_concealed_from_prompt():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "tests" / "golden").mkdir(parents=True)
        (td / "tests" / "golden" / "secret.py").write_text(
            "GOLDEN_SECRET_TOKEN = 1\n", encoding="utf-8"
        )
        proj = _FakeProject(
            td,
            "no code blocks here",
            config_extra={
                "orchestrator": {
                    "max_task_attempts": 1,
                    "golden_paths": ["tests/golden/"],
                }
            },
        )
        captured = {}

        def _cap(prompt, system_prompt=""):
            from my_project_orchestrator.llm.client import LLMResponse

            captured["text"] = (prompt or "") + (system_prompt or "")
            return LLMResponse(content="no code blocks here", finish_reason="stop")

        proj.llm_client.generate = _cap
        task = _make_task()
        task.context_files = ["tests/golden/secret.py"]
        task.processor_data["strategy"] = "surgical"
        MarkdownPlanExecutor().execute(task, proj, use_git_branch=False)
        assert "GOLDEN_SECRET_TOKEN" not in captured.get("text", "")


# ----------------------------------------------------------------
# Bounded continuation-on-truncation (plain generate path)
# ----------------------------------------------------------------


class _ScriptedLLMClient:
    """Returns a scripted sequence of (content, finish_reason) per generate()."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    @contextmanager
    def track_task(self, task_id):
        yield

    def generate(self, prompt, system_prompt=""):
        from my_project_orchestrator.llm.client import LLMResponse

        self.calls += 1
        content, finish = self._script[min(self.calls - 1, len(self._script) - 1)]
        return LLMResponse(content=content, finish_reason=finish)

    def generate_code(self, prompt, system_prompt=""):
        return self.generate(prompt, system_prompt).content


def _scripted_project(td, script, config_extra=None):
    proj = _FakeProject(td, "", config_extra)
    proj.llm_client = _ScriptedLLMClient(script)
    return proj


def test_truncated_response_continues_and_parses():
    # First call truncates mid-fence ("length"); the continuation finishes the
    # block ("stop"). The executor concatenates and the full edit applies.
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        head = "```python:new_mod.py\ndef alpha():\n    return 1\n"
        tail = "\ndef beta():\n    return 2\n```\n"
        proj = _scripted_project(td, [(head, "length"), (tail, "stop")])
        task = _make_task()
        task.processor_data["strategy"] = "surgical"
        task.processor_data["test_command"] = "true"
        task.files_to_create = ["new_mod.py"]
        result = MarkdownPlanExecutor().execute(task, proj, use_git_branch=False)

        assert result.status == "completed"
        assert proj.llm_client.calls == 2
        written = (td / "new_mod.py").read_text()
        assert "def alpha():" in written
        assert "def beta():" in written


def test_untruncated_response_makes_exactly_one_call():
    # A normal "stop" response must not trigger any continuation: exactly one
    # model call, and the returned text is the response verbatim.
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        proj = _scripted_project(td, [("This works. Implemented and done.", "stop")])
        e = MarkdownPlanExecutor()
        text, aborted = e._invoke_llm(proj, "p", "s")
        assert proj.llm_client.calls == 1
        assert text == "This works. Implemented and done."
        assert aborted is False


def test_max_continuations_zero_never_continues():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        proj = _scripted_project(
            td,
            [("partial", "length"), ("more", "stop")],
            config_extra={"orchestrator": {"max_continuations": 0}},
        )
        e = MarkdownPlanExecutor()
        text, _ = e._invoke_llm(proj, "p", "s")
        assert proj.llm_client.calls == 1
        assert text == "partial"


def test_continuation_is_bounded_by_cap():
    # A client that always truncates must stop after 1 + max_continuations calls.
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        proj = _scripted_project(
            td,
            [("x", "length")],
            config_extra={"orchestrator": {"max_continuations": 3}},
        )
        e = MarkdownPlanExecutor()
        text, _ = e._invoke_llm(proj, "p", "s")
        assert proj.llm_client.calls == 1 + 3
        assert text == "xxxx"


def test_is_truncated_helper():
    from my_project_orchestrator.task_executors.markdown_plan_executor import (
        _is_truncated,
    )

    assert _is_truncated("length")
    assert _is_truncated("max_tokens")
    assert _is_truncated("MAX_TOKENS")
    assert not _is_truncated("stop")
    assert not _is_truncated("")
    assert not _is_truncated(None)


# ----------------------------------------------------------------
# Adversarial critic (independent second component) — executor wiring
# ----------------------------------------------------------------


class _CriticFakeLLMClient:
    """Generation goes through generate(); the critic uses generate_code(), so a
    single client can drive the edit and return a separate critic verdict."""

    def __init__(self, edit_response: str, critic_verdict: str):
        self._edit_response = edit_response
        self._critic_verdict = critic_verdict
        self.generate_code_calls = 0
        self.critic_models = []
        self._active_model = None

    @contextmanager
    def track_task(self, task_id):
        yield

    @contextmanager
    def with_model(self, model):
        self._active_model = model
        try:
            yield self
        finally:
            self._active_model = None

    def generate(self, prompt, system_prompt=""):
        from my_project_orchestrator.llm.client import LLMResponse

        return LLMResponse(content=self._edit_response, finish_reason="stop")

    def generate_code(self, prompt, system_prompt=""):
        self.generate_code_calls += 1
        self.critic_models.append(self._active_model)
        return self._critic_verdict


def _run_critic_task(td, file_path, content, critic_verdict, orchestrator_cfg, critic_cfg=None):
    task = _make_task()
    task.processor_data["strategy"] = "surgical"
    task.processor_data["test_command"] = "true"
    task.files_to_modify = [file_path]
    proj = _FakeProject(td, _edit_response(file_path, content))
    proj.llm_client = _CriticFakeLLMClient(_edit_response(file_path, content), critic_verdict)
    proj.config["orchestrator"] = orchestrator_cfg
    if critic_cfg is not None:
        proj.config["critic"] = critic_cfg
    result = MarkdownPlanExecutor().execute(task, proj, use_git_branch=False)
    return result, proj


def test_critic_off_by_default_not_invoked():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        result, proj = _run_critic_task(
            td,
            "a.py",
            "x = 1\n",
            '{"approved": false, "objections": ["should not be consulted"]}',
            {"max_task_attempts": 1},  # adversarial_critic absent -> off
        )
        assert result.status == "completed"
        # Off path: the critic must never be called, so the rejection is ignored.
        assert proj.llm_client.generate_code_calls == 0


def test_critic_approve_applies_edit():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        result, proj = _run_critic_task(
            td,
            "a.py",
            "x = 1\n",
            '{"approved": true}',
            {"max_task_attempts": 2, "adversarial_critic": True},
        )
        assert result.status == "completed"
        assert proj.llm_client.generate_code_calls == 1
        assert (td / "a.py").read_text().strip() == "x = 1"


def test_critic_reject_then_defers_to_gates_after_bound():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Critic always rejects; max_rejections=1 means it rejects once (forcing a
        # regenerate) then defers to the real gates, which pass -> completed.
        result, proj = _run_critic_task(
            td,
            "a.py",
            "x = 1\n",
            '{"approved": false, "objections": ["missing edge case"]}',
            {
                "max_task_attempts": 3,
                "adversarial_critic": True,
                "critic_max_rejections": 1,
            },
        )
        assert result.status == "completed"
        # Consulted exactly once: the single allowed rejection, then it steps aside.
        assert proj.llm_client.generate_code_calls == 1
        assert (td / "a.py").read_text().strip() == "x = 1"


def test_critic_uses_independent_model_from_config():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        result, proj = _run_critic_task(
            td,
            "a.py",
            "x = 1\n",
            '{"approved": true}',
            {"max_task_attempts": 2, "adversarial_critic": True},
            critic_cfg={"model": "independent/critic-model"},
        )
        assert result.status == "completed"
        # The critique ran under the configured INDEPENDENT model, not the
        # generator's own — the whole point of the second component.
        assert proj.llm_client.critic_models == ["independent/critic-model"]


def test_build_critic_error_context_lists_objections():
    e = MarkdownPlanExecutor()
    ctx = e._build_critic_error_context(["no null check", "leaks a handle"])
    assert "no null check" in ctx
    assert "leaks a handle" in ctx
    assert "independent reviewer" in ctx.lower()


def test_judge_generate_uses_independent_model():
    # The LLM acceptance judge runs on judge.model when configured, not the
    # generator's own model — independence propagated to the post-impl judge.
    client = _CriticFakeLLMClient("", '{"approved": true}')

    class _Proj:
        def __init__(self, c):
            self.llm_client = c
            self.config = {"judge": {"model": "independent/judge"}}

    e = MarkdownPlanExecutor()
    out = e._judge_generate(_Proj(client), "is it ok?")
    assert out == '{"approved": true}'
    assert client.critic_models == ["independent/judge"]


def test_judge_generate_without_model_uses_generator():
    client = _CriticFakeLLMClient("", "PASS")

    class _Proj:
        def __init__(self, c):
            self.llm_client = c
            self.config = {}

    e = MarkdownPlanExecutor()
    out = e._judge_generate(_Proj(client), "is it ok?")
    assert out == "PASS"
    assert client.critic_models == [None]
