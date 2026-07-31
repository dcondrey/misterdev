"""Unit tests for ReportingMixin."""

import pytest
from unittest.mock import MagicMock, patch, call

from misterdev.core.execution.reporting_mixin import ReportingMixin


class _Orch(ReportingMixin):
    last_build_cost = 0.0


def _make_project(path=None):
    p = MagicMock()
    p.config = {}
    p.path = MagicMock()
    p.model_ledger = MagicMock()
    return p


def _make_report(completed=None, failed=None, deferred=None):
    r = MagicMock()
    r.completed_tasks = completed or []
    r.failed_tasks = failed or []
    r.deferred_tasks = deferred or []
    r.degraded_subsystems = []
    r.start_time = MagicMock()
    r.end_time = None
    return r


def _make_task(id_="t1", failure_text=""):
    t = MagicMock()
    t.id = id_
    t.processor_data = {"failure_text": failure_text} if failure_text else {}
    t.execution_history = []
    return t


# ---------------------------------------------------------------------------
# _task_failure_text
# ---------------------------------------------------------------------------


def test_task_failure_text_from_processor_data():
    task = _make_task(failure_text="merge conflict in foo.py")
    assert ReportingMixin._task_failure_text(task) == "merge conflict in foo.py"


def test_task_failure_text_from_execution_history():
    task = _make_task()
    result = MagicMock()
    result.message = "test failed"
    result.logs = "AssertionError"
    task.execution_history = [result]
    text = ReportingMixin._task_failure_text(task)
    assert "test failed" in text
    assert "AssertionError" in text


def test_task_failure_text_empty_when_nothing():
    task = _make_task()
    assert ReportingMixin._task_failure_text(task) == ""


def test_task_failure_text_prefers_processor_data():
    task = _make_task(failure_text="stored error")
    result = MagicMock()
    result.message = "other"
    result.logs = ""
    task.execution_history = [result]
    assert ReportingMixin._task_failure_text(task) == "stored error"


# ---------------------------------------------------------------------------
# _persist_learning
# ---------------------------------------------------------------------------


def test_persist_learning_records_cost():
    orch = _Orch()
    proj = _make_project()
    proj.llm_client.cumulative_usage.estimated_cost = 1.23
    report = _make_report()
    with (
        patch("misterdev.core.execution.reporting_mixin.FailureLog"),
        patch("misterdev.core.execution.reporting_mixin.SolvedTaskIndex"),
        patch.object(orch, "_record_env_learnings"),
        patch.object(orch, "_write_run_summary"),
    ):
        orch._persist_learning(proj, report)
    assert orch.last_build_cost == pytest.approx(1.23)


def test_persist_learning_degraded_on_failure_log_error():
    orch = _Orch()
    proj = _make_project()
    proj.llm_client.cumulative_usage.estimated_cost = 0.0
    report = _make_report()
    with (
        patch(
            "misterdev.core.execution.reporting_mixin.FailureLog",
            side_effect=OSError("disk full"),
        ),
        patch("misterdev.core.execution.reporting_mixin.SolvedTaskIndex"),
        patch.object(orch, "_record_env_learnings"),
        patch.object(orch, "_write_run_summary"),
    ):
        orch._persist_learning(proj, report)
    assert any("Failure logging" in s for s in report.degraded_subsystems)


def test_persist_learning_continues_after_one_failure():
    orch = _Orch()
    proj = _make_project()
    proj.llm_client.cumulative_usage.estimated_cost = 0.0
    report = _make_report()
    write_called = []
    with (
        patch("misterdev.core.execution.reporting_mixin.FailureLog"),
        patch(
            "misterdev.core.execution.reporting_mixin.SolvedTaskIndex",
            side_effect=OSError("boom"),
        ),
        patch.object(orch, "_record_env_learnings"),
        patch.object(
            orch, "_write_run_summary", side_effect=lambda *a: write_called.append(True)
        ),
    ):
        orch._persist_learning(proj, report)
    assert write_called  # _write_run_summary still ran despite SolvedTaskIndex error


# ---------------------------------------------------------------------------
# _record_env_learnings
# ---------------------------------------------------------------------------


def test_record_env_learnings_persists_setup_command():
    orch = _Orch()
    proj = _make_project()
    proj.env_settled_workers = None
    proj.env_base_workers = None
    fake_learnings = MagicMock()
    with (
        patch(
            "misterdev.core.execution.reporting_mixin.EnvLearnings.load",
            return_value=fake_learnings,
        ),
        patch(
            "misterdev.core.execution.reporting_mixin.worktree_setup_command",
            return_value="uv sync",
        ),
        patch(
            "misterdev.core.execution.reporting_mixin.worktree_healthcheck_command",
            return_value=None,
        ),
    ):
        orch._record_env_learnings(proj)
    assert fake_learnings.worktree_setup_command == "uv sync"
    fake_learnings.save.assert_called_once()


def test_record_env_learnings_clears_max_workers_when_recovered():
    orch = _Orch()
    proj = _make_project()
    proj.env_settled_workers = 4
    proj.env_base_workers = 4  # settled == base → no reduction → clear
    fake_learnings = MagicMock()
    with (
        patch(
            "misterdev.core.execution.reporting_mixin.EnvLearnings.load",
            return_value=fake_learnings,
        ),
        patch(
            "misterdev.core.execution.reporting_mixin.worktree_setup_command",
            return_value=None,
        ),
        patch(
            "misterdev.core.execution.reporting_mixin.worktree_healthcheck_command",
            return_value=None,
        ),
    ):
        orch._record_env_learnings(proj)
    assert fake_learnings.max_workers is None
