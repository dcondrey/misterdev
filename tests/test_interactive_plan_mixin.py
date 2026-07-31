"""Unit tests for InteractivePlanMixin."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from misterdev.core.execution.interactive_plan_mixin import InteractivePlanMixin
from misterdev.core.modes import BuildMode


class _Orch(InteractivePlanMixin):
    pass


def _make_project(path=None, has_git=True):
    p = MagicMock()
    p.config = {}
    mock_path = MagicMock(spec=Path)
    mock_path.__str__ = lambda self: str(path or "/fake/project")
    mock_path.__truediv__ = lambda self, other: (
        MagicMock(exists=lambda: has_git) if other == ".git" else MagicMock()
    )
    p.path = mock_path
    return p


# ---------------------------------------------------------------------------
# _working_tree_dirty
# ---------------------------------------------------------------------------


def test_working_tree_dirty_no_git_returns_empty():
    orch = _Orch()
    proj = _make_project(has_git=False)
    result = orch._working_tree_dirty(proj)
    assert result == ""


def test_working_tree_dirty_clean_tree_returns_empty():
    orch = _Orch()
    proj = _make_project(has_git=True)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        result = orch._working_tree_dirty(proj)
    assert result == ""


def test_working_tree_dirty_returns_summary():
    orch = _Orch()
    proj = _make_project(has_git=True)
    mock_git_dir = MagicMock()
    mock_git_dir.exists.return_value = True
    mock_path_obj = MagicMock()
    mock_path_obj.__truediv__ = lambda self, other: mock_git_dir
    with (
        patch(
            "misterdev.core.execution.interactive_plan_mixin.Path",
            return_value=mock_path_obj,
        ),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(
            returncode=0, stdout=" M misterdev/agent.py\n?? newfile.py\n"
        )
        result = orch._working_tree_dirty(proj)
    assert "2 file(s)" in result


def test_working_tree_dirty_subprocess_error_returns_empty():
    orch = _Orch()
    proj = _make_project(has_git=True)
    with patch("subprocess.run", side_effect=subprocess.SubprocessError("fail")):
        result = orch._working_tree_dirty(proj)
    assert result == ""


# ---------------------------------------------------------------------------
# _confirm
# ---------------------------------------------------------------------------


def test_confirm_yes_returns_true():
    orch = _Orch()
    with patch(
        "misterdev.core.execution.interactive_plan_mixin.Prompt.ask", return_value="y"
    ):
        assert orch._confirm("Proceed?") is True


def test_confirm_no_returns_false():
    orch = _Orch()
    with patch(
        "misterdev.core.execution.interactive_plan_mixin.Prompt.ask", return_value="n"
    ):
        assert orch._confirm("Proceed?") is False


def test_confirm_default_empty_returns_false():
    orch = _Orch()
    with patch(
        "misterdev.core.execution.interactive_plan_mixin.Prompt.ask", return_value=""
    ):
        assert orch._confirm("Proceed?") is False


# ---------------------------------------------------------------------------
# _choose_goal
# ---------------------------------------------------------------------------


def test_choose_goal_quit_returns_none():
    orch = _Orch()
    with patch(
        "misterdev.core.execution.interactive_plan_mixin.Prompt.ask", return_value="q"
    ):
        goal, mode = orch._choose_goal([])
    assert goal is None


def test_choose_goal_free_text_returns_text():
    orch = _Orch()
    with patch(
        "misterdev.core.execution.interactive_plan_mixin.Prompt.ask",
        return_value="fix the login bug",
    ):
        with patch(
            "misterdev.core.execution.interactive_plan_mixin.resolve_mode",
            return_value=BuildMode.DEBUG,
        ):
            goal, mode = orch._choose_goal([])
    assert goal == "fix the login bug"
    assert mode == BuildMode.DEBUG


def test_choose_goal_picks_rec_by_number():
    orch = _Orch()
    rec = MagicMock()
    rec.title = "Add tests"
    rec.work_type = "test"
    rec.rationale = "coverage low"
    with patch(
        "misterdev.core.execution.interactive_plan_mixin.Prompt.ask", return_value="1"
    ):
        goal, mode = orch._choose_goal([rec])
    assert goal == "Add tests"
    assert mode == BuildMode.SMART


def test_choose_goal_out_of_range_falls_back_to_text():
    orch = _Orch()
    rec = MagicMock()
    rec.title = "Add tests"
    rec.work_type = "test"
    rec.rationale = "coverage low"
    with patch(
        "misterdev.core.execution.interactive_plan_mixin.Prompt.ask", return_value="99"
    ):
        with patch(
            "misterdev.core.execution.interactive_plan_mixin.resolve_mode",
            return_value=BuildMode.SMART,
        ):
            goal, mode = orch._choose_goal([rec])
    assert goal == "99"


def test_work_type_modes_coverage():
    orch = _Orch()
    assert orch._WORK_TYPE_MODES["debug"] == BuildMode.DEBUG
    assert orch._WORK_TYPE_MODES["complete"] == BuildMode.COMPLETE
    assert orch._WORK_TYPE_MODES["feature"] == BuildMode.SMART
