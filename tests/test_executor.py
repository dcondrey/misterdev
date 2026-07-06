import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from misterdev.task_executors.markdown_plan_executor import (
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
from misterdev.core.models import Task
from misterdev.core.context.scratchpad import Scratchpad


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
        from misterdev.llm.client import LLMResponse

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
        self, description, target_files, max_symbols=30, ranker=None, exclude_files=None
    ):
        return ""

    def reference_sites(self, target_files, max_refs=80):
        return ""

    def invalidate(self):
        pass

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
    from misterdev.core.context.topography import TopographyEngine

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
    from misterdev.core.context.topography import TopographyEngine

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
    from misterdev.core.context.topography import TopographyEngine

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


def test_extract_acceptance_command_strips_trailing_prose_clause():
    # A command with an appended English clause (no period) reduces to the clean
    # command instead of a mangled shell string that fails good code.
    assert (
        _extract_acceptance_command(
            "python -m pytest -q' exits with code 0 and all 16 tests pass"
        )
        == "python -m pytest -q"
    )
    assert _extract_acceptance_command("run pytest and check the output") == "pytest"


def test_extract_acceptance_command_keeps_balanced_quoted_args():
    # A legitimate command with balanced quotes is NOT over-trimmed.
    assert _extract_acceptance_command('pytest -k "test_foo"') == 'pytest -k "test_foo"'


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


def test_critic_auto_enables_for_cross_cutting_categories():
    from types import SimpleNamespace

    ex = MarkdownPlanExecutor()
    auto = {"orchestrator": {"adversarial_critic": "auto"}}
    # auto: on for refactor/fix/integration, off for feature.
    assert ex._critic_enabled_for(
        SimpleNamespace(config=auto), SimpleNamespace(category="refactor")
    )
    assert not ex._critic_enabled_for(
        SimpleNamespace(config=auto), SimpleNamespace(category="feature")
    )
    # explicit True/False override the auto behavior.
    on = {"orchestrator": {"adversarial_critic": True}}
    assert ex._critic_enabled_for(
        SimpleNamespace(config=on), SimpleNamespace(category="feature")
    )
    off = {"orchestrator": {"adversarial_critic": False}}
    assert not ex._critic_enabled_for(
        SimpleNamespace(config=off), SimpleNamespace(category="refactor")
    )


def test_detect_dangling_references_flags_missed_caller():
    from types import SimpleNamespace
    from misterdev.core.context.topography.nodes import SymbolNode

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "insights.rs").write_text("fn analyze() { trope(); }\n")
        trope = SymbolNode("trope", "trope.rs", "function", 1, 3, "fn trope() {}")
        trope.incoming_calls = {"insights.rs:analyze"}
        analyze = SymbolNode("analyze", "insights.rs", "function", 1, 1, "")
        graph = SimpleNamespace(
            symbols={"trope.rs:trope": trope, "insights.rs:analyze": analyze}
        )
        project = SimpleNamespace(path=root, topography=SimpleNamespace(graph=graph))
        ex = MarkdownPlanExecutor()

        # Edit removes `trope` but leaves the caller in insights.rs untouched.
        flagged = ex._detect_dangling_references(project, {"trope.rs": "// removed\n"})
        assert flagged and "insights.rs:1" in flagged and "trope" in flagged

        # Edit that also updates the caller -> complete, nothing dangling.
        assert (
            ex._detect_dangling_references(
                project,
                {"trope.rs": "// removed\n", "insights.rs": "fn analyze() {}\n"},
            )
            is None
        )

        # Substring-only match (`tropes`) must NOT be flagged (word-boundary).
        (root / "insights.rs").write_text("fn analyze() { count_tropes(); }\n")
        assert (
            ex._detect_dangling_references(project, {"trope.rs": "// removed\n"})
            is None
        )


def test_acceptance_manifest_error_passes_through(monkeypatch):
    # A manifest error from the acceptance command means the command is
    # malformed (build/test already passed, so the manifest exists) — it must
    # NOT fail the task. This is the emathy `--manifest-path`-dropped bug.
    with tempfile.TemporaryDirectory() as td:
        proj = _FakeProject(Path(td), _edit_response("m.py", "x=1\n"))
        task = _make_task()
        task.acceptance_criteria = "cargo test -p emathy-core --lib"
        e = MarkdownPlanExecutor()
        monkeypatch.setattr(
            e,
            "_run_command",
            lambda *a, **k: (False, "error: could not find `Cargo.toml`"),
        )
        passed, _ = e._verify_acceptance(proj, task, True, False, 10)
        assert passed is True


def test_acceptance_real_test_failure_still_fails(monkeypatch):
    # A genuine (non-structural) command failure must still fail acceptance.
    with tempfile.TemporaryDirectory() as td:
        proj = _FakeProject(Path(td), _edit_response("m.py", "x=1\n"))
        task = _make_task()
        task.acceptance_criteria = "cargo test -p emathy-core --lib"
        e = MarkdownPlanExecutor()
        monkeypatch.setattr(
            e,
            "_run_command",
            lambda *a, **k: (False, "test result: FAILED. 1 passed; 2 failed"),
        )
        passed, _ = e._verify_acceptance(proj, task, True, False, 10)
        assert passed is False


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

from misterdev.task_executors.markdown_plan_executor import (  # noqa: E402
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
            from misterdev.llm.client import LLMResponse

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
        from misterdev.llm.client import LLMResponse

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
    from misterdev.task_executors.markdown_plan_executor import (
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
        from misterdev.llm.client import LLMResponse

        return LLMResponse(content=self._edit_response, finish_reason="stop")

    def generate_code(self, prompt, system_prompt=""):
        self.generate_code_calls += 1
        self.critic_models.append(self._active_model)
        return self._critic_verdict


def _run_critic_task(
    td, file_path, content, critic_verdict, orchestrator_cfg, critic_cfg=None
):
    task = _make_task()
    task.processor_data["strategy"] = "surgical"
    task.processor_data["test_command"] = "true"
    task.files_to_modify = [file_path]
    proj = _FakeProject(td, _edit_response(file_path, content))
    proj.llm_client = _CriticFakeLLMClient(
        _edit_response(file_path, content), critic_verdict
    )
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
                "reflection": False,
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


# ----------------------------------------------------------------
# Baseline-aware per-task test gate (incremental progress on a red baseline)
# ----------------------------------------------------------------


def test_gate_accepts_unit_cases():
    g = MarkdownPlanExecutor._gate_accepts
    assert g(True, "", 5) == (True, 0)  # green always passes
    assert g(False, "# fail 3", 0) == (False, None)  # baseline 0 -> strict
    assert g(False, "# tests 10\n# fail 3", 5) == (True, 3)  # improved
    assert g(False, "# tests 10\n# fail 5", 5) == (True, 5)  # not worse
    assert g(False, "# tests 10\n# fail 7", 5) == (False, 7)  # worse -> reject
    assert g(False, "no countable output", 5) == (False, None)  # unparseable -> strict


def _run_gated_task(td, test_command, baseline_failures):
    task = _make_task()
    task.processor_data["strategy"] = "surgical"
    task.processor_data["test_command"] = test_command
    task.files_to_modify = ["a.py"]
    proj = _FakeProject(
        td,
        _edit_response("a.py", "x = 1\n"),
        config_extra={
            "orchestrator": {
                "max_task_attempts": 1,
                "llm_acceptance_judge": False,
                "verify_acceptance": False,
            }
        },
    )
    proj.baseline_test_failures = baseline_failures
    return MarkdownPlanExecutor().execute(task, proj, use_git_branch=False)


def test_red_but_not_worse_task_completes():
    # The headline fix: on a 5-failure baseline, a task that leaves 3 failing
    # (an incremental fix) is accepted and completes instead of being discarded.
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "a.py").write_text("x = 0\n", encoding="utf-8")
        result = _run_gated_task(
            td, "echo '# tests 10'; echo '# fail 3'; exit 1", baseline_failures=5
        )
        assert result.status == "completed"


def test_worse_than_baseline_task_does_not_complete():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "a.py").write_text("x = 0\n", encoding="utf-8")
        result = _run_gated_task(
            td, "echo '# tests 10'; echo '# fail 7'; exit 1", baseline_failures=5
        )
        assert result.status != "completed"


def test_red_baseline_zero_is_strict():
    # With no baseline failures (green project), a failing suite still rejects.
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "a.py").write_text("x = 0\n", encoding="utf-8")
        result = _run_gated_task(
            td, "echo '# tests 10'; echo '# fail 1'; exit 1", baseline_failures=0
        )
        assert result.status != "completed"


class _Capturing(_FakeLLMClient):
    def __init__(self, response):
        super().__init__(response)
        self.prompts = []

    def generate(self, prompt, system_prompt=""):
        self.prompts.append(prompt)
        return super().generate(prompt, system_prompt)


def _seed_run(td, baseline_failures, baseline_output, error_template="fix the error"):
    (td / "a.py").write_text("x = 0\n", encoding="utf-8")
    task = _make_task()
    task.processor_data["strategy"] = "surgical"
    task.files_to_modify = ["a.py"]
    proj = _FakeProject(td, _edit_response("a.py", "x = 1\n"))
    proj.config["prompt_templates"]["error_correction_instruction"] = error_template
    proj.llm_client = _Capturing(_edit_response("a.py", "x = 1\n"))
    if baseline_failures:
        proj.baseline_test_failures = baseline_failures
        proj.baseline_test_output = baseline_output
    MarkdownPlanExecutor().execute(task, proj, use_git_branch=False)
    return proj.llm_client.prompts


def test_first_attempt_uses_error_template_when_seeded():
    # On a RED baseline, attempt 0 uses the error-correction template (which shows
    # failures) instead of the plain task template — so the model isn't blind.
    with tempfile.TemporaryDirectory() as td:
        prompts = _seed_run(Path(td), 1, "FAIL: something broke")
        assert prompts[0] == "fix the error"  # error_correction template selected


def test_green_baseline_uses_task_template():
    # No baseline failures -> attempt 0 uses the normal task template (unchanged).
    with tempfile.TemporaryDirectory() as td:
        prompts = _seed_run(Path(td), 0, "")
        assert prompts[0] == "do the task"  # task_completion template


def test_seed_content_reaches_prompt():
    # The actual failure text is substituted into the first prompt.
    with tempfile.TemporaryDirectory() as td:
        prompts = _seed_run(
            Path(td),
            1,
            "module does not provide export createRateLimiter",
            error_template="FAILURES:\n{error_logs}",
        )
        assert "createRateLimiter" in prompts[0]


# ----------------------------------------------------------------
# Multi-target gate routing (polyglot)
# ----------------------------------------------------------------


def test_target_routing_gates_in_target_files_with_target_commands():
    # A task whose files live under a target is gated by THAT target's commands.
    # Top-level build is "false" (would fail); the web target's is "true".
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "web").mkdir()
        (td / "web" / "x.ts").write_text("const x = 0;\n", encoding="utf-8")
        task = _make_task()
        task.processor_data["strategy"] = "surgical"
        task.files_to_modify = ["web/x.ts"]
        proj = _FakeProject(
            td,
            _edit_response("web/x.ts", "const x = 1;\n"),
            config_extra={
                "build_command": "false",
                "targets": [
                    {
                        "name": "web",
                        "path": "web",
                        "build_command": "true",
                        "test_command": "true",
                    }
                ],
                "orchestrator": {
                    "max_task_attempts": 1,
                    "verify_acceptance": False,
                    "llm_acceptance_judge": False,
                },
            },
        )
        result = MarkdownPlanExecutor().execute(task, proj, use_git_branch=False)
        assert result.status == "completed"  # routed to target's passing gate


def test_non_target_file_uses_top_level_build():
    # A file outside every target uses the top-level build ("false" -> fails).
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "other.py").write_text("x = 0\n", encoding="utf-8")
        task = _make_task()
        task.processor_data["strategy"] = "surgical"
        task.files_to_modify = ["other.py"]
        proj = _FakeProject(
            td,
            _edit_response("other.py", "x = 1\n"),
            config_extra={
                "build_command": "false",
                "targets": [{"name": "web", "path": "web", "build_command": "true"}],
                "orchestrator": {"max_task_attempts": 1},
            },
        )
        result = MarkdownPlanExecutor().execute(task, proj, use_git_branch=False)
        assert result.status != "completed"  # top-level "false" build fails


def test_target_gate_runs_in_target_directory():
    # The routed gate must execute in the TARGET's dir. Proof: a build command
    # that only succeeds when run from web/ (a marker file lives only there).
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "web").mkdir()
        (td / "web" / "x.ts").write_text("const x = 0;\n", encoding="utf-8")
        (td / "web" / "marker.txt").write_text("ok\n", encoding="utf-8")
        task = _make_task()
        task.processor_data["strategy"] = "surgical"
        task.files_to_modify = ["web/x.ts"]
        proj = _FakeProject(
            td,
            _edit_response("web/x.ts", "const x = 1;\n"),
            config_extra={
                "build_command": "false",
                "targets": [
                    {
                        "name": "web",
                        "path": "web",
                        "build_command": "test -f marker.txt",
                        "test_command": "true",
                    }
                ],
                "orchestrator": {
                    "max_task_attempts": 1,
                    "verify_acceptance": False,
                    "llm_acceptance_judge": False,
                },
            },
        )
        result = MarkdownPlanExecutor().execute(task, proj, use_git_branch=False)
        # Passes only if `test -f marker.txt` ran inside web/ (marker is not at root).
        assert result.status == "completed"


# ----------------------------------------------------------------
# Frontend test-file recognition + tamper protection (Swift / C# / Kotlin)
# ----------------------------------------------------------------


def test_is_test_file_frontend_languages():
    assert _is_test_file("clients/apple/Tests/EngineTests.swift")
    assert _is_test_file("ParserTest.kt")
    assert _is_test_file("Interop/EmathyEngineTests.cs")
    assert _is_test_file("clients/apple/Tests/Foo.swift")  # Tests/ dir (capital T)
    assert not _is_test_file("clients/apple/Sources/App/ContentView.swift")


def test_tamper_swift_test_removed_and_skip_added():
    before = (
        "import XCTest\n"
        "final class EngineTests: XCTestCase {\n"
        "  func testOpen() { XCTAssertEqual(open(), 1) }\n"
        "  func testClose() { XCTAssertEqual(close(), 0) }\n"
        "}\n"
    )
    # One test deleted -> tamper.
    after_removed = (
        "import XCTest\n"
        "final class EngineTests: XCTestCase {\n"
        "  func testOpen() { XCTAssertEqual(open(), 1) }\n"
        "}\n"
    )
    assert _diagnose_tampering(before, after_removed) is not None
    # A skip added -> tamper.
    after_skipped = before.replace(
        "func testClose() {", 'func testClose() throws { throw XCTSkip("x");'
    )
    assert _diagnose_tampering(before, after_skipped) is not None


def test_tamper_csharp_ignore_added():
    before = "[Fact]\npublic void Works() { Assert.Equal(1, F()); }\n"
    after = "[Fact]\n[Ignore]\npublic void Works() { Assert.Equal(1, F()); }\n"
    assert _diagnose_tampering(before, after) is not None


def test_frontend_test_additions_allowed():
    before = "[Fact]\npublic void A() { Assert.Equal(1, F()); }\n"
    after = (
        "[Fact]\npublic void A() { Assert.Equal(1, F()); }\n"
        "[Fact]\npublic void B() { Assert.Equal(2, G()); }\n"
    )
    assert _diagnose_tampering(before, after) is None  # pure addition is fine


def test_build_verified_task_completes_without_tests_or_certainty():
    # A typecheck-only frontend target: build passes, no test_command. A green
    # build is objective verification, so completion must NOT be blocked by a low
    # LLM certainty score (regression: web pilot tasks failed this way).
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "a.ts").write_text("const x = 0;\n", encoding="utf-8")
        task = _make_task()
        task.processor_data["strategy"] = "surgical"
        task.files_to_modify = ["a.ts"]
        proj = _FakeProject(
            td,
            _edit_response("a.ts", "const x = 1;\n"),
            config_extra={
                "build_command": "true",  # typecheck-style gate, passes
                "orchestrator": {
                    "max_task_attempts": 1,
                    "verify_acceptance": False,
                    "llm_acceptance_judge": False,
                    "certainty_threshold": 0.99,  # force certainty < threshold
                },
            },
        )
        result = MarkdownPlanExecutor().execute(task, proj, use_git_branch=False)
        assert result.status == "completed"


def test_unverified_low_certainty_still_refused_without_build():
    # The guard still holds when NOTHING verified the change: no test, no build,
    # low certainty -> refuse silent completion (task fails after its one attempt).
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "a.ts").write_text("const x = 0;\n", encoding="utf-8")
        task = _make_task()
        task.processor_data["strategy"] = "surgical"
        task.files_to_modify = ["a.ts"]
        proj = _FakeProject(
            td,
            _edit_response("a.ts", "const x = 1;\n"),
            config_extra={
                "orchestrator": {
                    "max_task_attempts": 1,
                    "certainty_threshold": 0.99,
                }
            },
        )
        result = MarkdownPlanExecutor().execute(task, proj, use_git_branch=False)
        assert result.status != "completed"


def test_full_file_fallback_after_repeated_apply_failures():
    # A SEARCH/REPLACE block whose SEARCH never matches -> repeated apply
    # failures -> the executor escalates the next attempt to a full-file rewrite
    # (no anchoring), breaking the stall instead of looping on the same failure.
    bad = (
        "```ts:a.ts\n<<<<<<< SEARCH\nDOES NOT EXIST IN FILE\n"
        "=======\nconst y = 9;\n>>>>>>> REPLACE\n```"
    )
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "a.ts").write_text("const x = 0;\n", encoding="utf-8")
        task = _make_task()
        task.processor_data["strategy"] = "surgical"
        task.files_to_modify = ["a.ts"]
        proj = _FakeProject(td, bad)
        proj.llm_client = _Capturing(bad)
        proj.config["orchestrator"] = {"max_task_attempts": 3, "reflection": False}
        proj.config["llm"] = {"use_tools": False}  # text SEARCH/REPLACE path
        MarkdownPlanExecutor().execute(task, proj, use_git_branch=False)
        ps = proj.llm_client.prompts
        assert len(ps) >= 3
        assert "SEARCH/REPLACE" in ps[0]  # first attempt: anchored format
        assert "COMPLETE," not in ps[0]
        assert "COMPLETE," in ps[2]  # third: full-file fallback engaged
