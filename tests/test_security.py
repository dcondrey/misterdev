import tempfile
import shlex
from pathlib import Path

from my_project_orchestrator.tools.git_tool import GitTool
from my_project_orchestrator.tools.file_io import FileIOTool


def test_git_branch_escaping():
    gt = GitTool({"name": "Git", "type": "git"})
    # Verify shlex.quote is used in all parameterized methods
    import inspect
    for method_name in ["branch_create", "branch_delete", "merge", "checkout",
                        "worktree_add", "worktree_remove", "add", "commit"]:
        src = inspect.getsource(getattr(gt, method_name))
        assert "shlex.quote" in src, f"{method_name} missing shlex.quote"


def test_shlex_quote_blocks_injection():
    dangerous = "; rm -rf /"
    safe = shlex.quote(dangerous)
    assert safe.startswith("'")
    assert "rm -rf" not in safe.split("'")[0]


def test_file_io_path_traversal():
    with tempfile.TemporaryDirectory() as td:
        class FP:
            path = Path(td)
        fio = FileIOTool({"name": "FileIO", "type": "file_io"})

        ok, msg = fio.execute(FP(), action="read", path="../../../etc/passwd")
        assert not ok
        assert "traversal" in msg.lower()


def test_file_io_path_traversal_write():
    with tempfile.TemporaryDirectory() as td:
        class FP:
            path = Path(td)
        fio = FileIOTool({"name": "FileIO", "type": "file_io"})

        ok, msg = fio.execute(FP(), action="write", path="../../evil.py", content="malicious")
        assert not ok
        assert "traversal" in msg.lower()


def test_file_io_normal_path_works():
    with tempfile.TemporaryDirectory() as td:
        class FP:
            path = Path(td)
        fio = FileIOTool({"name": "FileIO", "type": "file_io"})

        ok, msg = fio.execute(FP(), action="write", path="test.txt", content="hello")
        assert ok
        ok, content = fio.execute(FP(), action="read", path="test.txt")
        assert ok
        assert content == "hello"
