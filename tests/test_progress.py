import json
import tempfile
from pathlib import Path

from misterdev.core.execution.progress import ProgressTracker
from misterdev.utils.file_utils import orchestrator_state_file


def test_mark_and_check():
    with tempfile.TemporaryDirectory() as td:
        pt = ProgressTracker(Path(td))
        assert not pt.is_done("T-001")
        pt.mark_completed("T-001")
        assert pt.is_done("T-001")
        assert "T-001" in pt.completed


def test_persistence_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        pt1 = ProgressTracker(Path(td))
        pt1.mark_completed("T-001")
        pt1.mark_completed("T-002")
        pt1.mark_failed("T-003")

        pt2 = ProgressTracker(Path(td))
        assert pt2.is_done("T-001")
        assert pt2.is_done("T-002")
        assert "T-003" in pt2.failed
        assert not pt2.is_done("T-003")


def test_mark_completed_clears_failed():
    with tempfile.TemporaryDirectory() as td:
        pt = ProgressTracker(Path(td))
        pt.mark_failed("T-001")
        assert "T-001" in pt.failed
        pt.mark_completed("T-001")
        assert "T-001" not in pt.failed
        assert pt.is_done("T-001")


def test_completed_after_failed_is_only_completed():
    """Marking completed after failed leaves the task in exactly one terminal
    state: completed wins and the failed entry is cleared."""
    with tempfile.TemporaryDirectory() as td:
        pt = ProgressTracker(Path(td))
        pt.mark_failed("T-001")
        pt.mark_completed("T-001")
        assert "T-001" in pt.completed
        assert "T-001" not in pt.failed


def test_failed_after_completed_is_rejected():
    """The reverse is rejected: a task already completed is never re-listed as
    failed, so it stays only in completed (single terminal state)."""
    with tempfile.TemporaryDirectory() as td:
        pt = ProgressTracker(Path(td))
        pt.mark_completed("T-001")
        pt.mark_failed("T-001")
        assert "T-001" in pt.completed
        assert "T-001" not in pt.failed
        # And it survives a reload — the rejection was persisted, not just in-memory.
        assert "T-001" not in ProgressTracker(Path(td)).failed


def test_load_reconciles_poisoned_ledger():
    """An existing progress.json listing a task in BOTH completed and failed (the
    observed T002/T062a poisoning) self-heals on load: it is dropped from failed
    and the healed state is written back to disk."""
    with tempfile.TemporaryDirectory() as td:
        state = orchestrator_state_file(Path(td), "progress.json")
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(
            json.dumps(
                {
                    "completed": ["T002", "T062a", "T003"],
                    "failed": ["T002", "T062a", "T099"],
                    "hashes": {},
                }
            ),
            encoding="utf-8",
        )
        pt = ProgressTracker(Path(td))
        assert pt.completed == {"T002", "T062a", "T003"}
        assert pt.failed == {"T099"}  # poisoned entries dropped, real failure kept
        # Healed state was persisted, so a fresh load sees the clean ledger.
        on_disk = json.loads(state.read_text(encoding="utf-8"))
        assert set(on_disk["failed"]) == {"T099"}


def test_reset():
    with tempfile.TemporaryDirectory() as td:
        pt = ProgressTracker(Path(td))
        pt.mark_completed("T-001")
        pt.mark_failed("T-002")
        pt.reset()
        assert not pt.completed
        assert not pt.failed
        assert not pt.has_previous_run()


def test_has_previous_run():
    with tempfile.TemporaryDirectory() as td:
        pt = ProgressTracker(Path(td))
        assert not pt.has_previous_run()
        pt.mark_completed("T-001")
        assert pt.has_previous_run()


def test_summary():
    with tempfile.TemporaryDirectory() as td:
        pt = ProgressTracker(Path(td))
        pt.mark_completed("T-001")
        pt.mark_failed("T-002")
        assert "1 completed" in pt.summary()
        assert "1 failed" in pt.summary()
