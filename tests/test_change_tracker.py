import tempfile
import subprocess
from pathlib import Path

from my_project_orchestrator.core.context.change_tracker import ChangeTracker, TaskChange


def _init_git_repo(path: Path):
    subprocess.run("git init", shell=True, cwd=path, capture_output=True)
    subprocess.run("git add -A", shell=True, cwd=path, capture_output=True)
    subprocess.run(
        'git commit -m "init" --allow-empty', shell=True, cwd=path, capture_output=True
    )


def test_task_change_roundtrip():
    tc = TaskChange("T-001", ["src/lib.rs"], "+5/-2", 5, 2)
    d = tc.to_dict()
    tc2 = TaskChange.from_dict(d)
    assert tc2.task_id == "T-001"
    assert tc2.files == ["src/lib.rs"]
    assert tc2.additions == 5


def test_persistence():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ct1 = ChangeTracker(td)
        ct1.changes["T-001"] = TaskChange("T-001", ["a.py"], "diff", 10, 3)
        ct1._save()

        ct2 = ChangeTracker(td)
        assert "T-001" in ct2.changes
        assert ct2.changes["T-001"].additions == 10


def test_get_recent_changes_for_files():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ct = ChangeTracker(td)
        ct.changes["T-001"] = TaskChange(
            "T-001", ["src/lib.rs", "src/core/mod.rs"], "added pub mod", 5, 0
        )
        ct.changes["T-002"] = TaskChange(
            "T-002", ["src/posting.rs"], "new file", 100, 0
        )
        ct.changes["T-003"] = TaskChange(
            "T-003", ["src/core/mod.rs"], "added posting import", 1, 0
        )

        result = ct.get_recent_changes_for_files(["src/core/mod.rs"])
        assert "T-003" in result
        assert "T-001" in result
        assert "T-002" not in result


def test_get_recent_changes_empty():
    with tempfile.TemporaryDirectory() as td:
        ct = ChangeTracker(Path(td))
        assert ct.get_recent_changes_for_files(["nonexistent.rs"]) == ""


def test_record_task_changes_in_git():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "file.txt").write_text("original")
        _init_git_repo(td)
        (td / "file.txt").write_text("modified")
        subprocess.run("git add -A", shell=True, cwd=td, capture_output=True)
        subprocess.run(
            'git commit -m "task change"', shell=True, cwd=td, capture_output=True
        )

        ct = ChangeTracker(td)
        change = ct.record_task_changes("T-001", ["file.txt"])
        assert change.task_id == "T-001"
        assert change.additions >= 1 or change.deletions >= 1


def test_max_entries_cap():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ct = ChangeTracker(td)
        for i in range(20):
            ct.changes[f"T-{i:03d}"] = TaskChange(
                f"T-{i:03d}", ["shared.rs"], f"change {i}", 1, 0
            )

        result = ct.get_recent_changes_for_files(["shared.rs"], max_entries=3)
        assert result.count("###") == 3
