import tempfile
from pathlib import Path

from my_project_orchestrator.config import ConfigManager
from my_project_orchestrator.core.task import TaskManager


def _setup_project(td, task_files):
    td = Path(td)
    (td / "project.yaml").write_text("name: test\n")
    dp = td / "devplan"
    dp.mkdir()
    for name, content in task_files.items():
        (dp / name).write_text(content)

    cfg = ConfigManager().load_project_config(td)

    class FP:
        path = td
        config = cfg

    return TaskManager(FP())


def test_discover_basic():
    with tempfile.TemporaryDirectory() as td:
        tm = _setup_project(td, {
            "001-foo.md": "---\nstatus: pending\n---\nDo foo\n",
        })
        tasks = tm.discover_tasks()
        assert len(tasks) == 1
        assert tasks[0].id == "001-foo"
        assert tasks[0].status == "pending"


def test_skips_docs_without_status():
    with tempfile.TemporaryDirectory() as td:
        tm = _setup_project(td, {
            "001-task.md": "---\nstatus: pending\n---\nReal task\n",
            "ARCHITECTURE.md": "# Architecture\nDocs only.\n",
            "README.md": "---\ntitle: Readme\n---\nNo status.\n",
        })
        tasks = tm.discover_tasks()
        assert len(tasks) == 1
        assert tasks[0].id == "001-task"


def test_dependency_resolution():
    with tempfile.TemporaryDirectory() as td:
        tm = _setup_project(td, {
            "001-posting.md": "---\nstatus: pending\ndepends_on: []\n---\nT1\n",
            "006-indexed.md": '---\nstatus: pending\ndepends_on: ["001"]\n---\nT6\n',
        })
        tm.discover_tasks()
        t6 = tm.tasks["006-indexed"]
        assert t6.dependencies == ["001-posting"]


def test_pending_excludes_completed():
    with tempfile.TemporaryDirectory() as td:
        tm = _setup_project(td, {
            "001-done.md": "---\nstatus: completed\n---\nDone\n",
            "002-pending.md": "---\nstatus: pending\n---\nPending\n",
        })
        tm.discover_tasks()
        pending = tm.get_pending_tasks()
        assert len(pending) == 1
        assert pending[0].id == "002-pending"


def test_dedup_on_rediscovery():
    with tempfile.TemporaryDirectory() as td:
        tm = _setup_project(td, {
            "001-foo.md": "---\nstatus: pending\n---\nFoo\n",
        })
        tm.discover_tasks()
        tm.discover_tasks()
        assert len(tm.tasks) == 1


def test_sorted_order():
    with tempfile.TemporaryDirectory() as td:
        tm = _setup_project(td, {
            "003-c.md": "---\nstatus: pending\n---\nC\n",
            "001-a.md": "---\nstatus: pending\n---\nA\n",
            "002-b.md": "---\nstatus: pending\n---\nB\n",
        })
        tasks = tm.discover_tasks()
        assert [t.id for t in tasks] == ["001-a", "002-b", "003-c"]


def test_frontmatter_fields():
    with tempfile.TemporaryDirectory() as td:
        tm = _setup_project(td, {
            "001-foo.md": "---\nstatus: pending\ntitle: Implement Foo\ncategory: core\ncomplexity: large\nfiles_to_modify:\n  - src/foo.rs\ncontext_files:\n  - src/lib.rs\n---\nDescription\n",
        })
        tm.discover_tasks()
        t = tm.tasks["001-foo"]
        assert t.title == "Implement Foo"
        assert t.category == "core"
        assert t.complexity == "large"
        assert t.files_to_modify == ["src/foo.rs"]
        assert t.context_files == ["src/lib.rs"]
