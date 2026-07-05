import tempfile
import shlex
from pathlib import Path

from my_project_orchestrator.tools.git_tool import GitTool
from my_project_orchestrator.tools.file_io import FileIOTool


def test_git_branch_escaping():
    gt = GitTool({"name": "Git", "type": "git"})
    # Verify shlex.quote is used in all parameterized methods
    import inspect

    for method_name in [
        "branch_create",
        "branch_delete",
        "merge",
        "checkout",
        "worktree_add",
        "worktree_remove",
        "add",
        "commit",
    ]:
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

        ok, msg = fio.execute(
            FP(), action="write", path="../../evil.py", content="malicious"
        )
        assert not ok
        assert "traversal" in msg.lower()


def test_file_io_delete_root_blocked():
    # An empty/"."/"./" path resolves to the project root and previously passed
    # the traversal guard, so delete would rmtree the entire project.
    for bad in ("", ".", "./"):
        with tempfile.TemporaryDirectory() as td:
            sentinel = Path(td) / "keep.txt"
            sentinel.write_text("keep")

            class FP:
                path = Path(td)

            fio = FileIOTool({"name": "FileIO", "type": "file_io"})
            ok, msg = fio.execute(FP(), action="delete", path=bad)
            assert not ok, f"delete of root path {bad!r} should be refused"
            assert "root" in msg.lower()
            assert sentinel.exists(), (
                "project contents must survive a root-delete attempt"
            )
            assert Path(td).exists()


def test_file_io_delete_git_dir_blocked():
    # Deleting .git would destroy the version history the orchestrator's
    # rollback/bisect paths depend on; it must be refused (the root guard alone
    # does not cover it).
    for bad in (".git", ".git/config", "./.git"):
        with tempfile.TemporaryDirectory() as td:
            git = Path(td) / ".git"
            (git).mkdir()
            (git / "config").write_text("[core]\n")

            class FP:
                path = Path(td)

            fio = FileIOTool({"name": "FileIO", "type": "file_io"})
            ok, msg = fio.execute(FP(), action="delete", path=bad)
            assert not ok, f"delete of {bad!r} should be refused"
            assert "git" in msg.lower()
            assert (git / "config").exists(), ".git must survive the attempt"


def test_file_io_read_is_capped():
    # A model-requested read of an over-large in-project file is bounded (with a
    # truncation marker), not loaded whole into memory / the LLM context.
    from my_project_orchestrator.utils import file_utils

    with tempfile.TemporaryDirectory() as td:
        big = Path(td) / "huge.txt"
        big.write_text("A" * (file_utils._MAX_READ_CHARS + 5000))

        class FP:
            path = Path(td)

        fio = FileIOTool({"name": "FileIO", "type": "file_io"})
        ok, content = fio.execute(FP(), action="read", path="huge.txt")
        assert ok
        assert len(content) <= file_utils._MAX_READ_CHARS + 100
        assert "truncated" in content


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
