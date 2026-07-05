import subprocess
import tempfile
from pathlib import Path

from misterdev.task_executors.markdown_plan_executor import (
    MarkdownPlanExecutor,
)


def _init_git_repo(path: Path):
    subprocess.run("git init", shell=True, cwd=path, capture_output=True)
    subprocess.run("git add -A", shell=True, cwd=path, capture_output=True)
    subprocess.run(
        'git commit -m "init" --allow-empty', shell=True, cwd=path, capture_output=True
    )


def test_is_git_repo():
    executor = MarkdownPlanExecutor()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        class FP:
            path = td

        assert not executor._is_git_repo(FP())
        _init_git_repo(td)
        assert executor._is_git_repo(FP())


def test_create_and_get_branch():
    executor = MarkdownPlanExecutor()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "file.txt").write_text("hello")
        _init_git_repo(td)

        class FP:
            path = td

        base = executor._get_current_branch(FP())
        assert base is not None

        ok = executor._create_task_branch(FP(), "task/T-001")
        assert ok

        current = executor._get_current_branch(FP())
        assert current == "task/T-001"

        # Switch back
        executor._git(FP(), f"git checkout {base}")
        assert executor._get_current_branch(FP()) == base


def test_abort_deletes_branch():
    executor = MarkdownPlanExecutor()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "file.txt").write_text("original")
        _init_git_repo(td)

        class FP:
            path = td

        base = executor._get_current_branch(FP())
        executor._create_task_branch(FP(), "task/T-002")

        # Make a change on the task branch
        (td / "file.txt").write_text("modified")
        executor._git(FP(), "git add -A")
        executor._git(FP(), 'git commit -m "task change"')

        # Abort
        executor._abort_task(FP(), "task/T-002", base, None)

        assert executor._get_current_branch(FP()) == base
        assert (td / "file.txt").read_text() == "original"

        # Branch should be deleted
        ok, output = executor._git(FP(), "git branch")
        assert "task/T-002" not in output


def test_fallback_without_git():
    executor = MarkdownPlanExecutor()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "test.py").write_text("x = 1")

        class FP:
            path = td

        assert not executor._is_git_repo(FP())
        # Snapshot should still work
        snapshot = executor._snapshot_files(FP(), ["test.py"])
        assert snapshot["test.py"] == "x = 1"
