"""Unit tests for PipelineMixin — structural helpers only.

_run_pipeline requires deep collaborator setup; these tests cover the
simpler helpers that can be exercised with minimal mocking.
"""

import pytest
from unittest.mock import MagicMock, patch, call

from misterdev.core.execution.pipeline_mixin import PipelineMixin
from misterdev.core.models import Task


class _Orch(PipelineMixin):
    last_build_succeeded = True
    last_build_cost = 0.0

    def _persist_learning(self, *a, **kw):
        pass

    def _capture_head(self, *a, **kw):
        return None

    def _run_goal_check(self, *a, **kw):
        pass

    def _execute_tasks(self, *a, **kw):
        pass

    def _project_file_map(self, *a, **kw):
        return {}

    def _resolve_targets(self, *a, **kw):
        return []

    def _staging_hint(self, *a, **kw):
        return None

    def _verify_completeness_claims(self, *a, **kw):
        pass

    def _generate_spec(self, *a, **kw):
        return "spec text"

    def _build_fix_spec(self, *a, **kw):
        return "fix spec"

    def _container_engine(self, *a, **kw):
        return None

    def _validate_targets(self, *a, **kw):
        return []

    def _maybe_rollback_regression(self, *a, **kw):
        pass

    def _setup_env(self, *a, **kw):
        return None

    def _analyze(self, *a, **kw):
        return MagicMock()

    def _confirm(self, *a, **kw):
        return True

    def _learning_embedder(self, *a, **kw):
        return None


def _make_task(id_, deps=None):
    t = MagicMock(spec=Task)
    t.id = id_
    t.dependencies = deps or []
    t.files_to_modify = []
    t.files_to_create = []
    t.complexity = "normal"
    t.category = "feature"
    t.description = "do something"
    t.title = id_
    t.processor_data = {}
    return t


def _make_project():
    p = MagicMock()
    p.config = {}
    p.path = MagicMock()
    p.llm_client.cumulative_usage.estimated_cost = 0.0
    p.baseline_test_failures = 0
    p.baseline_test_output = ""
    return p


# ---------------------------------------------------------------------------
# _requirements_preflight
# ---------------------------------------------------------------------------


def test_requirements_preflight_returns_true_when_proceed():
    orch = _Orch()
    proj = _make_project()
    result = orch._requirements_preflight(proj, [], proceed=True)
    assert result is True


def test_requirements_preflight_returns_true_when_setting_off():
    orch = _Orch()
    proj = _make_project()
    with patch(
        "misterdev.core.execution.pipeline_mixin.get_setting", return_value=False
    ):
        result = orch._requirements_preflight(proj, [], proceed=False)
    assert result is True


def test_requirements_preflight_degrades_on_import_error():
    orch = _Orch()
    proj = _make_project()
    with (
        patch("misterdev.core.execution.pipeline_mixin.get_setting", return_value=True),
        patch(
            "builtins.__import__",
            side_effect=lambda name, *a, **kw: (
                (_ for _ in ()).throw(ImportError("no module"))
                if "requirements" in name
                else __import__(name, *a, **kw)
            ),
        ),
    ):
        result = orch._requirements_preflight(proj, [], proceed=False)
    assert result is True


# ---------------------------------------------------------------------------
# _inject_task_context
# ---------------------------------------------------------------------------


def test_inject_task_context_populates_processor_data():
    orch = _Orch()
    task = _make_task("t1")
    task.dependencies = ["dep1"]
    task.files_to_modify = ["a.py"]
    task.files_to_create = []

    contracts = MagicMock()
    contracts.get_contracts_for_task.return_value = {"dep1": "iface"}
    changes = MagicMock()
    changes.get_recent_changes_for_files.return_value = ["change1"]
    strategy = MagicMock()
    strategy.select_best_strategy.return_value = "approach"
    proj = _make_project()

    orch._inject_task_context(task, contracts, changes, strategy, proj)

    assert task.processor_data["interface_contracts"] == {"dep1": "iface"}
    assert task.processor_data["recent_changes"] == ["change1"]
    assert task.processor_data["strategy"] == "approach"


# ---------------------------------------------------------------------------
# _print_execution_plan
# ---------------------------------------------------------------------------


def test_print_execution_plan_no_crash_empty():
    orch = _Orch()
    orch._print_execution_plan([])


def test_print_execution_plan_orders_by_dep():
    orch = _Orch()
    t1 = _make_task("t1")
    t2 = _make_task("t2", deps=["t1"])
    output_lines = []
    with patch("misterdev.core.execution.pipeline_mixin._console") as mock_console:
        mock_console.print.side_effect = lambda msg, *a, **kw: output_lines.append(
            str(msg)
        )
        orch._print_execution_plan([t1, t2])
    joined = " ".join(output_lines)
    assert "t1" in joined
    assert "t2" in joined


def test_print_execution_plan_detects_deadlock():
    orch = _Orch()
    t1 = _make_task("t1", deps=["t2"])
    t2 = _make_task("t2", deps=["t1"])
    with patch("misterdev.core.execution.pipeline_mixin._console") as mock_console:
        orch._print_execution_plan([t1, t2])
    calls = [str(c) for c in mock_console.print.call_args_list]
    assert any("deadlock" in c.lower() or "Dependency" in c for c in calls)


# ---------------------------------------------------------------------------
# _print_rerun_status
# ---------------------------------------------------------------------------


def test_print_rerun_status_shows_run_and_skip():
    orch = _Orch()
    t1 = _make_task("t1")
    t2 = _make_task("t2")
    progress = MagicMock()
    progress.needs_rerun.side_effect = lambda tid, _hash: tid == "t1"
    progress.is_done.return_value = False

    with patch("misterdev.core.execution.pipeline_mixin._console") as mock_console:
        orch._print_rerun_status([t1, t2], progress, "/path", force=False)

    calls = [str(c) for c in mock_console.print.call_args_list]
    joined = " ".join(calls)
    assert "t1" in joined
    assert "t2" in joined


def test_print_rerun_status_force_runs_all():
    orch = _Orch()
    tasks = [_make_task(f"t{i}") for i in range(3)]
    progress = MagicMock()
    progress.needs_rerun.return_value = False

    with patch("misterdev.core.execution.pipeline_mixin._console") as mock_console:
        orch._print_rerun_status(tasks, progress, "/path", force=True)

    calls = " ".join(str(c) for c in mock_console.print.call_args_list)
    assert "forced" in calls


# ---------------------------------------------------------------------------
# _halt_on_budget
# ---------------------------------------------------------------------------


def test_halt_on_budget_no_report_returns_message():
    orch = _Orch()
    proj = _make_project()
    result = orch._halt_on_budget(proj, None, Exception("cap hit"))
    assert "budget" in result.lower() or "halted" in result.lower()
    assert orch.last_build_succeeded is False


def test_halt_on_budget_with_report_saves_and_returns_markdown():
    orch = _Orch()
    proj = _make_project()
    report = MagicMock()
    report.key_decisions = []
    report.to_markdown.return_value = "## Build Report"

    result = orch._halt_on_budget(proj, report, Exception("$5 cap"))

    assert result == "## Build Report"
    report.finalize.assert_called_once()
    report.save.assert_called_once()
    assert orch.last_build_succeeded is False
