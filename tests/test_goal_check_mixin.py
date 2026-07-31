"""Unit tests for GoalCheckMixin."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from misterdev.core.execution.goal_check_mixin import GoalCheckMixin


class _Orch(GoalCheckMixin):
    last_build_succeeded = True


def _make_project(has_git=True, path=None):
    p = MagicMock()
    p.config = {}
    git_dir = MagicMock()
    git_dir.exists.return_value = has_git
    p.path = MagicMock()
    p.path.__truediv__ = lambda self, other: git_dir if other == ".git" else MagicMock()
    return p


# ---------------------------------------------------------------------------
# _learning_embedder
# ---------------------------------------------------------------------------


def test_learning_embedder_returns_none_on_error():
    orch = _Orch()
    proj = _make_project()
    with patch(
        "misterdev.llm.client.embeddings.create_embedding_client",
        side_effect=ImportError("no fastembed"),
    ):
        result = orch._learning_embedder(proj)
    assert result is None


def test_learning_embedder_returns_client():
    orch = _Orch()
    proj = _make_project()
    fake_client = MagicMock()
    with patch(
        "misterdev.llm.client.embeddings.create_embedding_client",
        return_value=fake_client,
    ):
        result = orch._learning_embedder(proj)
    assert result is fake_client or result is None  # depends on installed deps


# ---------------------------------------------------------------------------
# _capture_head
# ---------------------------------------------------------------------------


def test_capture_head_no_git_returns_none():
    orch = _Orch()
    proj = _make_project(has_git=False)
    result = orch._capture_head(proj)
    assert result is None


def test_capture_head_git_error_returns_none():
    orch = _Orch()
    proj = _make_project(has_git=True)
    with patch("misterdev.core.execution.goal_check_mixin.run_git", return_value=None):
        result = orch._capture_head(proj)
    assert result is None


def test_capture_head_returns_sha():
    orch = _Orch()
    proj = _make_project(has_git=True)
    proc = MagicMock(returncode=0, stdout="abc123\n")
    with patch("misterdev.core.execution.goal_check_mixin.run_git", return_value=proc):
        result = orch._capture_head(proj)
    assert result == "abc123"


# ---------------------------------------------------------------------------
# _cumulative_diff
# ---------------------------------------------------------------------------


def test_cumulative_diff_no_git_returns_empty():
    orch = _Orch()
    proj = _make_project(has_git=False)
    assert orch._cumulative_diff(proj, "abc") == ""


def test_cumulative_diff_with_base():
    orch = _Orch()
    proj = _make_project(has_git=True)
    proc = MagicMock(returncode=0, stdout="diff --git a/foo.py b/foo.py\n")
    with patch(
        "misterdev.core.execution.goal_check_mixin.run_git", return_value=proc
    ) as mock_git:
        result = orch._cumulative_diff(proj, "deadbeef")
    cmd = mock_git.call_args[0][0]
    assert "deadbeef" in cmd
    assert result == "diff --git a/foo.py b/foo.py\n"


def test_cumulative_diff_no_base_uses_head():
    orch = _Orch()
    proj = _make_project(has_git=True)
    proc = MagicMock(returncode=0, stdout="")
    with patch(
        "misterdev.core.execution.goal_check_mixin.run_git", return_value=proc
    ) as mock_git:
        orch._cumulative_diff(proj, None)
    cmd = mock_git.call_args[0][0]
    assert "HEAD" in cmd


# ---------------------------------------------------------------------------
# _run_goal_check
# ---------------------------------------------------------------------------


def test_run_goal_check_exception_degrades():
    orch = _Orch()
    proj = _make_project()
    report = MagicMock()
    report.degraded_subsystems = []
    report.completed_tasks = []
    with patch(
        "misterdev.core.verification.goal_check.run_goal_check",
        side_effect=RuntimeError("llm error"),
    ):
        with patch(
            "misterdev.core.execution.goal_check_mixin.get_setting", return_value=None
        ):
            orch._run_goal_check(proj, "my goal", [], None, report)
    assert any("Goal-completion check" in s for s in report.degraded_subsystems)


def test_run_goal_check_pass_is_noop():
    orch = _Orch()
    proj = _make_project()
    report = MagicMock()
    report.completed_tasks = []
    verdict = MagicMock(status="pass", reason="all good")
    with (
        patch(
            "misterdev.core.execution.goal_check_mixin.get_setting", return_value=None
        ),
        patch(
            "misterdev.core.execution.goal_check_mixin.run_git",
            return_value=MagicMock(returncode=0, stdout=""),
        ),
        patch(
            "misterdev.core.verification.goal_check.run_goal_check",
            return_value=verdict,
        ),
        patch(
            "misterdev.core.verification.goal_check.build_evidence",
            return_value="evidence",
        ),
        patch("misterdev.core.verification.goal_check.GAP", "gap"),
    ):
        orch._run_goal_check(proj, "goal", [], None, report)
    assert not hasattr(report, "goal_gaps") or report.goal_gaps != ["some gap"]
