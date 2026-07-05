import tempfile
from pathlib import Path

from misterdev.core.execution.progress import ProgressTracker


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
