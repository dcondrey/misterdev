import tempfile
from pathlib import Path

from my_project_orchestrator.task_executors.markdown_plan_executor import (
    MarkdownPlanExecutor, _detect_language, _LANG_MAP,
)
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
