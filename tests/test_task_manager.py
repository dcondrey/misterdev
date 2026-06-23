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
        tm = _setup_project(
            td,
            {
                "001-foo.md": "---\nstatus: pending\n---\nDo foo\n",
            },
        )
        tasks = tm.discover_tasks()
        assert len(tasks) == 1
        assert tasks[0].id == "001-foo"
        assert tasks[0].status == "pending"


def test_skips_docs_without_status():
    with tempfile.TemporaryDirectory() as td:
        tm = _setup_project(
            td,
            {
                "001-task.md": "---\nstatus: pending\n---\nReal task\n",
                "ARCHITECTURE.md": "# Architecture\nDocs only.\n",
                "README.md": "---\ntitle: Readme\n---\nNo status.\n",
            },
        )
        tasks = tm.discover_tasks()
        assert len(tasks) == 1
        assert tasks[0].id == "001-task"


def test_dependency_resolution():
    with tempfile.TemporaryDirectory() as td:
        tm = _setup_project(
            td,
            {
                "001-posting.md": "---\nstatus: pending\ndepends_on: []\n---\nT1\n",
                "006-indexed.md": '---\nstatus: pending\ndepends_on: ["001"]\n---\nT6\n',
            },
        )
        tm.discover_tasks()
        t6 = tm.tasks["006-indexed"]
        assert t6.dependencies == ["001-posting"]


def test_pending_excludes_completed():
    with tempfile.TemporaryDirectory() as td:
        tm = _setup_project(
            td,
            {
                "001-done.md": "---\nstatus: completed\n---\nDone\n",
                "002-pending.md": "---\nstatus: pending\n---\nPending\n",
            },
        )
        tm.discover_tasks()
        pending = tm.get_pending_tasks()
        assert len(pending) == 1
        assert pending[0].id == "002-pending"


def test_dedup_on_rediscovery():
    with tempfile.TemporaryDirectory() as td:
        tm = _setup_project(
            td,
            {
                "001-foo.md": "---\nstatus: pending\n---\nFoo\n",
            },
        )
        tm.discover_tasks()
        tm.discover_tasks()
        assert len(tm.tasks) == 1


def test_sorted_order():
    with tempfile.TemporaryDirectory() as td:
        tm = _setup_project(
            td,
            {
                "003-c.md": "---\nstatus: pending\n---\nC\n",
                "001-a.md": "---\nstatus: pending\n---\nA\n",
                "002-b.md": "---\nstatus: pending\n---\nB\n",
            },
        )
        tasks = tm.discover_tasks()
        assert [t.id for t in tasks] == ["001-a", "002-b", "003-c"]


def test_frontmatter_fields():
    with tempfile.TemporaryDirectory() as td:
        tm = _setup_project(
            td,
            {
                "001-foo.md": "---\nstatus: pending\ntitle: Implement Foo\ncategory: core\ncomplexity: large\nfiles_to_modify:\n  - src/foo.rs\ncontext_files:\n  - src/lib.rs\n---\nDescription\n",
            },
        )
        tm.discover_tasks()
        t = tm.tasks["001-foo"]
        assert t.title == "Implement Foo"
        assert t.category == "core"
        assert t.complexity == "large"
        assert t.files_to_modify == ["src/foo.rs"]
        assert t.context_files == ["src/lib.rs"]


def test_discover_missing_devplan_returns_empty():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "project.yaml").write_text("name: test\n")
        cfg = ConfigManager().load_project_config(td)

        class FP:
            path = td
            config = cfg

        assert TaskManager(FP()).discover_tasks() == []


def test_dependency_prefix_resolution_and_string_form():
    with tempfile.TemporaryDirectory() as td:
        tm = _setup_project(
            td,
            {
                "001-base.md": "---\nstatus: pending\n---\nbase\n",
                # depends_on given as a bare string short-id -> coerced + resolved
                "002-feat.md": "---\nstatus: pending\ndepends_on: '001'\n---\nfeat\n",
            },
        )
        tm.discover_tasks()
        assert tm.tasks["002-feat"].dependencies == ["001-base"]


def test_unresolved_dependency_kept_verbatim():
    with tempfile.TemporaryDirectory() as td:
        tm = _setup_project(
            td,
            {"001-a.md": "---\nstatus: pending\ndepends_on:\n  - 999\n---\na\n"},
        )
        tm.discover_tasks()
        assert tm.tasks["001-a"].dependencies == ["999"]


def test_file_overlap_adds_implicit_dependency():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "project.yaml").write_text(
            "name: test\norchestrator:\n  auto_detect_dependencies: true\n"
        )
        dp = td / "devplan"
        dp.mkdir()
        (dp / "001-a.md").write_text(
            "---\nstatus: pending\nfiles_to_modify:\n  - src/x.rs\n---\na\n"
        )
        (dp / "002-b.md").write_text(
            "---\nstatus: pending\nfiles_to_modify:\n  - src/x.rs\n---\nb\n"
        )
        cfg = ConfigManager().load_project_config(td)

        class FP:
            path = td
            config = cfg

        tm = TaskManager(FP())
        tm.discover_tasks()
        assert "001-a" in tm.tasks["002-b"].dependencies


def test_update_status_persists_and_skips_unknown():
    with tempfile.TemporaryDirectory() as td:
        tm = _setup_project(td, {"001-a.md": "---\nstatus: pending\n---\nwork\n"})
        tm.discover_tasks()
        tm.update_task_status("001-a", "completed")
        assert tm.tasks["001-a"].status == "completed"
        # persisted back to the markdown front-matter
        text = (Path(tm.project.path) / "devplan" / "001-a.md").read_text()
        assert "completed" in text
        # unknown id is a no-op (no raise)
        tm.update_task_status("nope", "completed")
