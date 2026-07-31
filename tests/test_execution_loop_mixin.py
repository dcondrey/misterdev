"""Unit tests for ExecutionLoopMixin — structural edge cases only.

The full wave loop requires deep collaborator setup; these tests target the
pure-logic paths (dep-deferral, budget exhaustion, empty task list) that can
be exercised with minimal mocking.
"""

import pytest
from unittest.mock import MagicMock, patch

from misterdev.core.execution.execution_loop_mixin import ExecutionLoopMixin
from misterdev.core.models import Task


class _Orch(ExecutionLoopMixin):
    def _suite_failures(self, *a, **kw):
        return 0

    def _failing_ids_from_output(self, *a, **kw):
        return None

    def _interactive_prompt(self, *a, **kw):
        return "proceed"

    def _apply_wave_tuning(self, *a, **kw):
        pass

    def _execute_parallel(self, *a, **kw):
        return []

    def _task_failure_text(self, task):
        return ""

    def _integration_gate(self, *a, **kw):
        return []

    def _integration_gate_targets(self, *a, **kw):
        return []

    def _wave_infra_count(self, results):
        return 0


def _make_project():
    p = MagicMock()
    p.config = {
        "orchestrator": {
            "max_consecutive_failures": 3,
            "max_workers": 2,
            "skip_satisfied_tasks": False,
            "adaptive_concurrency": False,
            "integration_gate": False,
            "max_cost_per_task": None,
            "golden_command": None,
            "adaptive_infra_threshold": 1,
            "adaptive_timeout_factor": 1.5,
            "adaptive_max_timeout_factor": 3.0,
        },
        "build": {"build_timeout": 120, "test_timeout": 90},
    }
    p.task_manager.tasks = {}
    p.baseline_test_failures = 0
    p.baseline_test_output = ""
    p.llm_client.cumulative_usage.estimated_cost = 0.0
    p.llm_client.task_cost_exceeded = None
    return p


def _make_flags(parallel=False, no_rollback=True, interactive=False, no_verify=False):
    f = MagicMock()
    f.parallel = parallel
    f.no_rollback = no_rollback
    f.interactive = interactive
    f.no_verify = no_verify
    return f


def _make_report():
    r = MagicMock()
    r.completed_tasks = []
    r.failed_tasks = []
    r.deferred_tasks = []
    r.key_decisions = []
    r.assessment.structure.test_command = None
    return r


def _make_task(id_, deps=None):
    t = MagicMock(spec=Task)
    t.id = id_
    t.dependencies = deps or []
    t.files_to_modify = []
    t.files_to_create = []
    t.complexity = "normal"
    t.category = "feature"
    t.description = "do something"
    t.processor_data = {}
    t.execution_history = []
    return t


# ---------------------------------------------------------------------------
# Empty task list
# ---------------------------------------------------------------------------


def test_execute_tasks_empty_list_no_crash():
    orch = _Orch()
    proj = _make_project()
    report = _make_report()
    with (
        patch("misterdev.core.execution.execution_loop_mixin.ProgressTracker"),
        patch("misterdev.core.execution.execution_loop_mixin.MarkdownPlanExecutor"),
        patch("misterdev.core.execution.execution_loop_mixin.Scratchpad"),
        patch("misterdev.core.execution.execution_loop_mixin.RealTimeAligner"),
        patch("misterdev.core.execution.execution_loop_mixin.ContractRegistry"),
        patch("misterdev.core.execution.execution_loop_mixin.ChangeTracker"),
        patch("misterdev.core.execution.execution_loop_mixin.StrategyOptimizer"),
        patch(
            "misterdev.core.execution.execution_loop_mixin.get_setting",
            side_effect=lambda cfg, *keys: {
                ("orchestrator", "max_workers"): 2,
                ("orchestrator", "max_consecutive_failures"): 3,
            }.get(keys),
        ),
        patch(
            "misterdev.core.execution.execution_loop_mixin._budget_exhausted",
            return_value=False,
        ),
        patch(
            "misterdev.core.execution.execution_loop_mixin._combine_commands",
            return_value=None,
        ),
    ):
        orch._execute_tasks([], proj, _make_flags(), report)
    assert report.deferred_tasks == []


# ---------------------------------------------------------------------------
# All deps failed → tasks deferred
# ---------------------------------------------------------------------------


def test_execute_tasks_failed_dep_defers_dependents():
    orch = _Orch()
    proj = _make_project()
    report = _make_report()

    parent = _make_task("p1")
    child = _make_task("c1", deps=["p1"])

    mock_progress = MagicMock()
    mock_progress.completed = []
    mock_progress.needs_rerun.return_value = True

    mock_executor = MagicMock()
    result = MagicMock(status="failed", logs="error", message="failed")
    mock_executor.execute.return_value = result
    mock_executor._run_command.return_value = (True, "")

    with (
        patch(
            "misterdev.core.execution.execution_loop_mixin.ProgressTracker",
            return_value=mock_progress,
        ),
        patch(
            "misterdev.core.execution.execution_loop_mixin.MarkdownPlanExecutor",
            return_value=mock_executor,
        ),
        patch("misterdev.core.execution.execution_loop_mixin.Scratchpad"),
        patch("misterdev.core.execution.execution_loop_mixin.RealTimeAligner"),
        patch("misterdev.core.execution.execution_loop_mixin.ContractRegistry"),
        patch("misterdev.core.execution.execution_loop_mixin.ChangeTracker"),
        patch("misterdev.core.execution.execution_loop_mixin.StrategyOptimizer"),
        patch(
            "misterdev.core.execution.execution_loop_mixin.get_setting",
            side_effect=lambda cfg, *keys: {
                ("orchestrator", "max_workers"): 2,
                ("orchestrator", "max_consecutive_failures"): 3,
            }.get(keys),
        ),
        patch(
            "misterdev.core.execution.execution_loop_mixin._budget_exhausted",
            return_value=False,
        ),
        patch(
            "misterdev.core.execution.execution_loop_mixin._combine_commands",
            return_value=None,
        ),
    ):
        orch._execute_tasks([parent, child], proj, _make_flags(), report)

    deferred_ids = {t.id for t in report.deferred_tasks}
    assert "c1" in deferred_ids


# ---------------------------------------------------------------------------
# Budget exhausted at start → all remaining deferred
# ---------------------------------------------------------------------------


def test_execute_tasks_budget_exhausted_defers_all():
    orch = _Orch()
    proj = _make_project()
    report = _make_report()
    tasks = [_make_task("t1"), _make_task("t2")]

    mock_progress = MagicMock()
    mock_progress.completed = []
    mock_progress.needs_rerun.return_value = True
    mock_executor = MagicMock()
    mock_executor._run_command.return_value = (True, "")

    with (
        patch(
            "misterdev.core.execution.execution_loop_mixin.ProgressTracker",
            return_value=mock_progress,
        ),
        patch(
            "misterdev.core.execution.execution_loop_mixin.MarkdownPlanExecutor",
            return_value=mock_executor,
        ),
        patch("misterdev.core.execution.execution_loop_mixin.Scratchpad"),
        patch("misterdev.core.execution.execution_loop_mixin.RealTimeAligner"),
        patch("misterdev.core.execution.execution_loop_mixin.ContractRegistry"),
        patch("misterdev.core.execution.execution_loop_mixin.ChangeTracker"),
        patch("misterdev.core.execution.execution_loop_mixin.StrategyOptimizer"),
        patch(
            "misterdev.core.execution.execution_loop_mixin.get_setting",
            side_effect=lambda cfg, *keys: {
                ("orchestrator", "max_workers"): 2,
                ("orchestrator", "max_consecutive_failures"): 3,
            }.get(keys),
        ),
        patch(
            "misterdev.core.execution.execution_loop_mixin._budget_exhausted",
            return_value=True,
        ),
        patch(
            "misterdev.core.execution.execution_loop_mixin._combine_commands",
            return_value=None,
        ),
    ):
        orch._execute_tasks(tasks, proj, _make_flags(), report)

    deferred_ids = {t.id for t in report.deferred_tasks}
    assert "t1" in deferred_ids or "t2" in deferred_ids
    assert any("budget" in d.lower() for d in report.key_decisions)
