"""Tests for configurable orchestrator limits via project.yaml orchestrator: key.

Verifies:
- DEFAULT_CONFIG contains the orchestrator section with correct defaults.
- agent._execute_tasks reads max_consecutive_failures from project.config.
- agent._execute_parallel reads max_workers from project.config.
- ContextBudget accepts a custom max_tokens constructor argument.
"""

import concurrent.futures
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from my_project_orchestrator.core.config import DEFAULT_CONFIG
from my_project_orchestrator.core.context_budget import ContextBudget


# ---------------------------------------------------------------------------
# DEFAULT_CONFIG
# ---------------------------------------------------------------------------


class TestDefaultConfig:
    """Verify DEFAULT_CONFIG has the expected orchestrator section."""

    def test_orchestrator_key_exists(self):
        assert "orchestrator" in DEFAULT_CONFIG

    def test_max_consecutive_failures_default(self):
        assert DEFAULT_CONFIG["orchestrator"]["max_consecutive_failures"] == 3

    def test_max_workers_default(self):
        assert DEFAULT_CONFIG["orchestrator"]["max_workers"] == 4

    def test_context_budget_tokens_default(self):
        assert DEFAULT_CONFIG["orchestrator"]["context_budget_tokens"] == 100000

    def test_max_task_attempts_default(self):
        assert DEFAULT_CONFIG["orchestrator"]["max_task_attempts"] == 3

    def test_count_values_are_positive_integers(self):
        # The numeric tuning knobs must be positive ints. Budget-driven keys
        # ("auto"), booleans, and the float certainty_threshold are intentional
        # and excluded.
        int_keys = (
            "max_consecutive_failures",
            "max_workers",
            "context_budget_tokens",
            "max_task_attempts",
        )
        cfg = DEFAULT_CONFIG["orchestrator"]
        for key in int_keys:
            value = cfg[key]
            assert isinstance(value, int) and value > 0, (
                f"orchestrator.{key} should be a positive int, got {value!r}"
            )

    def test_hardened_defaults_opted_in(self):
        cfg = DEFAULT_CONFIG["orchestrator"]
        assert cfg["max_build_iterations"] == "auto"
        assert cfg["max_cost_per_task"] == "auto"
        assert cfg["verify_acceptance"] is True
        assert cfg["llm_acceptance_judge"] is True
        assert cfg["allow_test_edits"] is False
        assert cfg["certainty_threshold"] == 0.5


# ---------------------------------------------------------------------------
# ContextBudget – custom max_tokens
# ---------------------------------------------------------------------------


class TestContextBudgetCustomTokens:
    """ContextBudget already accepts max_tokens; verify the parameter works."""

    def test_default_max_tokens(self):
        budget = ContextBudget()
        assert budget.max_tokens == 100000

    def test_custom_max_tokens(self):
        budget = ContextBudget(max_tokens=150000)
        assert budget.max_tokens == 150000

    def test_available_reflects_custom_max_tokens(self):
        budget = ContextBudget(max_tokens=150000, reserved_tokens=8000)
        assert budget.available == 142000

    def test_configured_value_from_default_config(self):
        """Constructing with the DEFAULT_CONFIG value should work."""
        tokens = DEFAULT_CONFIG["orchestrator"]["context_budget_tokens"]
        budget = ContextBudget(max_tokens=tokens)
        assert budget.max_tokens == tokens

    def test_larger_budget_allows_more_content(self):
        """A larger budget should not truncate content that a smaller one would."""
        content = "x" * 3500  # ~1000 tokens
        small_budget = ContextBudget(max_tokens=500, reserved_tokens=0)
        large_budget = ContextBudget(max_tokens=150000, reserved_tokens=0)

        small_budget.set("section", content, priority=1)
        large_budget.set("section", content, priority=1)

        small_result = small_budget.allocate()
        large_result = large_budget.allocate()

        assert len(large_result["section"]) >= len(small_result["section"])


# ---------------------------------------------------------------------------
# agent._execute_parallel – max_workers from config
# ---------------------------------------------------------------------------


class TestExecuteParallelMaxWorkers:
    """_execute_parallel should use orchestrator.max_workers from project.config."""

    def _make_orchestrator(self):
        from my_project_orchestrator.agent import ProjectOrchestrator

        return ProjectOrchestrator()

    def _make_project(self, orchestrator_cfg: dict) -> MagicMock:
        project = MagicMock()
        project.config = {"orchestrator": orchestrator_cfg}
        return project

    def _make_tasks(self, n: int):
        tasks = []
        for i in range(n):
            t = MagicMock()
            t.id = f"t{i}"
            tasks.append(t)
        return tasks

    def test_uses_configured_max_workers(self):
        orchestrator = self._make_orchestrator()
        project = self._make_project({"max_workers": 8})
        executor = MagicMock()
        tasks = self._make_tasks(3)  # 3 tasks, max_workers=8 → min(3,8)=3

        with patch(
            "my_project_orchestrator.agent.concurrent.futures.ThreadPoolExecutor"
        ) as MockPool:
            mock_ctx = MagicMock()
            MockPool.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            MockPool.return_value.__exit__ = MagicMock(return_value=False)
            mock_ctx.submit.return_value = MagicMock()

            with patch(
                "my_project_orchestrator.agent.concurrent.futures.as_completed",
                return_value=[],
            ):
                orchestrator._execute_parallel(tasks, executor, project)

            MockPool.assert_called_once_with(max_workers=3)

    def test_caps_at_task_count(self):
        """max_workers=8 but only 2 tasks → ThreadPoolExecutor(max_workers=2)."""
        orchestrator = self._make_orchestrator()
        project = self._make_project({"max_workers": 8})
        executor = MagicMock()
        tasks = self._make_tasks(2)

        with patch(
            "my_project_orchestrator.agent.concurrent.futures.ThreadPoolExecutor"
        ) as MockPool:
            mock_ctx = MagicMock()
            MockPool.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            MockPool.return_value.__exit__ = MagicMock(return_value=False)
            mock_ctx.submit.return_value = MagicMock()

            with patch(
                "my_project_orchestrator.agent.concurrent.futures.as_completed",
                return_value=[],
            ):
                orchestrator._execute_parallel(tasks, executor, project)

            MockPool.assert_called_once_with(max_workers=2)

    def test_default_max_workers_when_no_orchestrator_config(self):
        """Falls back to 4 when orchestrator key is absent."""
        orchestrator = self._make_orchestrator()
        project = MagicMock()
        project.config = {}  # No orchestrator key
        executor = MagicMock()
        tasks = self._make_tasks(6)  # 6 tasks, default max_workers=4 → min(6,4)=4

        with patch(
            "my_project_orchestrator.agent.concurrent.futures.ThreadPoolExecutor"
        ) as MockPool:
            mock_ctx = MagicMock()
            MockPool.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            MockPool.return_value.__exit__ = MagicMock(return_value=False)
            mock_ctx.submit.return_value = MagicMock()

            with patch(
                "my_project_orchestrator.agent.concurrent.futures.as_completed",
                return_value=[],
            ):
                orchestrator._execute_parallel(tasks, executor, project)

            MockPool.assert_called_once_with(max_workers=4)

    def test_default_max_workers_when_orchestrator_config_empty(self):
        """Falls back to 4 when orchestrator dict exists but max_workers is absent."""
        orchestrator = self._make_orchestrator()
        project = self._make_project({})  # orchestrator key present but empty
        executor = MagicMock()
        tasks = self._make_tasks(10)  # 10 tasks, default max_workers=4 → min(10,4)=4

        with patch(
            "my_project_orchestrator.agent.concurrent.futures.ThreadPoolExecutor"
        ) as MockPool:
            mock_ctx = MagicMock()
            MockPool.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            MockPool.return_value.__exit__ = MagicMock(return_value=False)
            mock_ctx.submit.return_value = MagicMock()

            with patch(
                "my_project_orchestrator.agent.concurrent.futures.as_completed",
                return_value=[],
            ):
                orchestrator._execute_parallel(tasks, executor, project)

            MockPool.assert_called_once_with(max_workers=4)


# ---------------------------------------------------------------------------
# agent._execute_parallel – disjoint-file partitioning in shared mode
# ---------------------------------------------------------------------------


class TestExecuteParallelDisjoint:
    """Shared mode runs only disjoint-file tasks in one concurrent batch."""

    def _make_orchestrator(self):
        from my_project_orchestrator.agent import ProjectOrchestrator

        return ProjectOrchestrator()

    def _make_project(self):
        project = MagicMock()
        # Explicit shared mode → no worktree promotion; non-git project.
        project.config = {"orchestrator": {"parallel_mode": "shared"}}
        project.path = MagicMock()
        (project.path / ".git").exists.return_value = False
        return project

    def _make_task(self, tid, modify=None, create=None):
        t = MagicMock()
        t.id = tid
        t.files_to_modify = list(modify or [])
        t.files_to_create = list(create or [])
        return t

    def test_overlapping_tasks_not_in_same_batch(self):
        orchestrator = self._make_orchestrator()
        project = self._make_project()

        # t0 & t1 share no files (disjoint → concurrent).
        # t2 overlaps t0 on a.py (must be serialized, not submitted to pool).
        t0 = self._make_task("t0", modify=["a.py"])
        t1 = self._make_task("t1", modify=["b.py"])
        t2 = self._make_task("t2", modify=["a.py", "c.py"])
        tasks = [t0, t1, t2]

        executor = MagicMock()
        executor.execute.return_value = MagicMock()

        submitted = []

        def fake_submit(fn, task, proj, use_git_branch=False):
            submitted.append(task)
            fut = MagicMock()
            fut.result.return_value = MagicMock()
            return fut

        with patch(
            "my_project_orchestrator.agent.concurrent.futures.ThreadPoolExecutor"
        ) as MockPool:
            mock_ctx = MagicMock()
            MockPool.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            MockPool.return_value.__exit__ = MagicMock(return_value=False)
            mock_ctx.submit.side_effect = fake_submit

            with patch(
                "my_project_orchestrator.agent.concurrent.futures.as_completed",
                side_effect=lambda d: list(d),
            ):
                results = orchestrator._execute_parallel(tasks, executor, project)

            # Only the disjoint pair was submitted concurrently.
            assert {t.id for t in submitted} == {"t0", "t1"}
            # Pool sized for the concurrent group (2), not all 3 tasks.
            MockPool.assert_called_once_with(max_workers=2)

        # t2 ran serially via the executor directly (not through the pool).
        serial_ids = {c.args[0].id for c in executor.execute.call_args_list}
        assert serial_ids == {"t2"}
        # Every task is represented in the returned tuples.
        assert {t.id for t, _, _ in results} == {"t0", "t1", "t2"}


# ---------------------------------------------------------------------------
# agent._execute_tasks – max_consecutive_failures from config
# ---------------------------------------------------------------------------


class TestExecuteTasksMaxConsecutiveFailures:
    """_execute_tasks should abort after the configured number of consecutive failures."""

    def _run_execute_tasks_with_config(
        self, orchestrator_cfg: dict, num_tasks: int = 5
    ):
        """Helper: run _execute_tasks with all-failing tasks and return the report mock."""
        from my_project_orchestrator.agent import ProjectOrchestrator
        from my_project_orchestrator.core.modes import BuildFlags

        orchestrator = ProjectOrchestrator()

        project = MagicMock()
        project.config = {"orchestrator": orchestrator_cfg, "language": "python"}
        project.path = Path("/tmp/fake_proj")
        # This suite exercises the consecutive-failure limit, not cost caps. A
        # bare MagicMock client returns truthy for task_cost_exceeded, which
        # would spuriously trip the (now default-"auto") per-task cap branch.
        project.llm_client.task_cost_exceeded.return_value = False

        failed_result = MagicMock()
        failed_result.status = "failed"

        report = MagicMock()
        report.completed_tasks = []
        report.failed_tasks = []
        report.deferred_tasks = []
        report.assessment = MagicMock()
        report.assessment.summary.return_value = "summary"

        tasks = []
        for i in range(num_tasks):
            t = MagicMock()
            t.id = f"t{i}"
            t.dependencies = []
            t.files_to_modify = []
            t.files_to_create = []
            t.processor_data = {}
            t.execution_history = []
            tasks.append(t)

        with (
            patch("my_project_orchestrator.agent.Scratchpad"),
            patch("my_project_orchestrator.agent.RealTimeAligner"),
            patch("my_project_orchestrator.agent.ContractRegistry"),
            patch("my_project_orchestrator.agent.ProgressTracker") as MockProgress,
            patch("my_project_orchestrator.agent.ChangeTracker"),
            patch("my_project_orchestrator.agent.StrategyOptimizer") as MockStrategy,
            patch("my_project_orchestrator.agent.MarkdownPlanExecutor") as MockExecutor,
        ):
            mock_progress = MockProgress.return_value
            mock_progress.completed = []
            mock_progress.is_done.return_value = False

            mock_strategy = MockStrategy.return_value
            mock_strategy.select_best_strategy.return_value = "iterative"

            mock_executor_instance = MockExecutor.return_value
            mock_executor_instance.execute.return_value = failed_result

            flags = BuildFlags()
            orchestrator._execute_tasks(tasks, project, flags, report)

        return report, mock_executor_instance

    def test_aborts_after_configured_limit(self):
        """With max_consecutive_failures=2, execution stops after 2 failures."""
        report, executor = self._run_execute_tasks_with_config(
            {"max_consecutive_failures": 2}, num_tasks=5
        )
        # Should have stopped after 2 failures; not all 5 tasks attempted
        total_attempted = len(report.failed_tasks) + len(report.completed_tasks)
        assert total_attempted <= 2

    def test_default_limit_when_no_config(self):
        """Without orchestrator config, falls back to default of 3."""
        from my_project_orchestrator.agent import ProjectOrchestrator
        from my_project_orchestrator.core.modes import BuildFlags

        orchestrator = ProjectOrchestrator()

        project = MagicMock()
        project.config = {}  # No orchestrator key
        project.path = Path("/tmp/fake_proj")
        # Not a cost-cap test; stop the bare MagicMock client from tripping the
        # default-"auto" per-task cap branch.
        project.llm_client.task_cost_exceeded.return_value = False

        failed_result = MagicMock()
        failed_result.status = "failed"

        report = MagicMock()
        report.completed_tasks = []
        report.failed_tasks = []
        report.deferred_tasks = []
        report.assessment = MagicMock()
        report.assessment.summary.return_value = "summary"

        tasks = []
        for i in range(10):
            t = MagicMock()
            t.id = f"t{i}"
            t.dependencies = []
            t.files_to_modify = []
            t.files_to_create = []
            t.processor_data = {}
            t.execution_history = []
            tasks.append(t)

        with (
            patch("my_project_orchestrator.agent.Scratchpad"),
            patch("my_project_orchestrator.agent.RealTimeAligner"),
            patch("my_project_orchestrator.agent.ContractRegistry"),
            patch("my_project_orchestrator.agent.ProgressTracker") as MockProgress,
            patch("my_project_orchestrator.agent.ChangeTracker"),
            patch("my_project_orchestrator.agent.StrategyOptimizer") as MockStrategy,
            patch("my_project_orchestrator.agent.MarkdownPlanExecutor") as MockExecutor,
        ):
            mock_progress = MockProgress.return_value
            mock_progress.completed = []
            mock_progress.is_done.return_value = False

            mock_strategy = MockStrategy.return_value
            mock_strategy.select_best_strategy.return_value = "iterative"

            mock_executor_instance = MockExecutor.return_value
            mock_executor_instance.execute.return_value = failed_result

            flags = BuildFlags()
            orchestrator._execute_tasks(tasks, project, flags, report)

        # Default is 3; should stop after 3 failures
        total_attempted = len(report.failed_tasks) + len(report.completed_tasks)
        assert total_attempted <= 3

    def test_higher_limit_allows_more_failures(self):
        """max_consecutive_failures=5 allows more failures before aborting."""
        report_2, _ = self._run_execute_tasks_with_config(
            {"max_consecutive_failures": 2}, num_tasks=10
        )
        report_5, _ = self._run_execute_tasks_with_config(
            {"max_consecutive_failures": 5}, num_tasks=10
        )

        attempted_2 = len(report_2.failed_tasks) + len(report_2.completed_tasks)
        attempted_5 = len(report_5.failed_tasks) + len(report_5.completed_tasks)

        assert attempted_5 >= attempted_2


# ---------------------------------------------------------------------------
# orchestrator.max_build_iterations – convergence cap default
# ---------------------------------------------------------------------------


class TestMaxBuildIterationsConfig:
    """The convergence loop reads orchestrator.max_build_iterations, defaulting
    to 1 so existing single-pass behavior is preserved when the key is absent."""

    def test_absent_key_defaults_to_auto_dynamic(self):
        # With no max_build_iterations key the default is "auto": a failing gate
        # is re-attempted (not a single pass). It still terminates here via the
        # no-progress guard once the identical failure repeats.
        from tests.test_orchestrator_fixes import _run_convergence_pipeline_with_cfg

        report, exec_calls, decompose_calls = _run_convergence_pipeline_with_cfg(
            gate_sequence=[(False, ["broke"])], orchestrator_cfg={}
        )
        assert exec_calls == 2  # iterated once more, not a single pass
        assert decompose_calls == 2  # baseline + one targeted fix re-decompose
        assert any("no progress" in d.lower() for d in report.key_decisions)

    def test_explicit_higher_cap_enables_iteration(self):
        from tests.test_orchestrator_fixes import _run_convergence_pipeline_with_cfg

        report, exec_calls, decompose_calls = _run_convergence_pipeline_with_cfg(
            gate_sequence=[(False, ["broke"]), (True, [])],
            orchestrator_cfg={"max_build_iterations": 3},
        )
        assert exec_calls == 2
        assert decompose_calls == 2  # baseline + one fix re-decompose


# ---------------------------------------------------------------------------
# Structural guard: every config knob must be wired, or this fails.
# Catches the recurring "define a config key, then ignore it / shadow it with a
# hardcoded constant" class of bug at CI time instead of in production.
# ---------------------------------------------------------------------------


def _accessed_config_keys():
    """All string literals used as a dict key via .get()/.pop()/[...] in source.

    A config knob that is honored is always read through one of these from a
    config dict, so a key absent from this set is dead or shadowed. (Keys read
    via a variable rather than a literal are not captured; acceptable.)
    """
    import ast
    from pathlib import Path

    pkg = Path(__file__).resolve().parent.parent / "my_project_orchestrator"
    keys: set[str] = set()
    for py in pkg.rglob("*.py"):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            # dict.get("key") / dict.pop("key")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("get", "pop")
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                keys.add(node.args[0].value)
            # get_setting(config, section, "key") and
            # get_section_setting(section, section_dict, "key") both carry the
            # key as the 3rd positional arg.
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("get_setting", "get_section_setting")
                and len(node.args) >= 3
                and isinstance(node.args[2], ast.Constant)
                and isinstance(node.args[2].value, str)
            ):
                keys.add(node.args[2].value)
            # dict["key"]
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                keys.add(node.slice.value)
    return keys


def test_every_tuning_config_key_is_wired():
    # The knob sections prone to orphaning. prompt_templates/tools are accessed
    # dynamically and excluded.
    accessed = _accessed_config_keys()
    dead = []
    for section in ("build", "orchestrator", "llm"):
        for key in DEFAULT_CONFIG.get(section, {}):
            if key not in accessed:
                dead.append(f"{section}.{key}")
    assert not dead, (
        "Config keys defined in DEFAULT_CONFIG but never read from a config "
        f"dict (dead or shadowed by a hardcoded constant): {dead}"
    )


# ---------------------------------------------------------------------------
# Typed config schema: DEFAULT_CONFIG is generated from dataclasses, and
# project.yaml keys are validated against them.
# ---------------------------------------------------------------------------


def test_default_config_generated_from_schema():
    from dataclasses import asdict
    from my_project_orchestrator.config import (
        DEFAULT_CONFIG,
        BuildSettings,
        OrchestratorSettings,
        LLMSettings,
    )

    # Each section must equal its schema's defaults — proving one source of truth.
    assert DEFAULT_CONFIG["build"] == asdict(BuildSettings())
    assert DEFAULT_CONFIG["orchestrator"] == asdict(OrchestratorSettings())
    assert DEFAULT_CONFIG["llm"] == asdict(LLMSettings())


def test_warn_unknown_keys_flags_typos_only():
    from my_project_orchestrator.config import warn_unknown_keys

    unknown = warn_unknown_keys(
        {
            "build": {"buildtimeout": 300, "max_tasks": 5},  # one typo, one valid
            "orchestrator": {"enable_ab_mcts": True},  # valid
        }
    )
    assert unknown == ["build.buildtimeout"]


def test_config_manager_warns_on_unknown_yaml_key(tmp_path, caplog):
    import logging
    from my_project_orchestrator.config import ConfigManager

    (tmp_path / "project.yaml").write_text("build:\n  buildtimeout: 300\n")
    with caplog.at_level(logging.WARNING):
        cfg = ConfigManager().load_project_config(tmp_path)
    assert any("build.buildtimeout" in r.message for r in caplog.records)
    # The typo is ignored; the real key keeps its schema default.
    assert cfg["build"]["build_timeout"] == 120


def test_parallel_mode_auto_isolates_via_worktrees_on_git_repo(tmp_path):
    # Regression guard: parallel_mode "auto" (the default) must use worktrees on
    # a git repo. This broke silently once parallel_mode gained a DEFAULT_CONFIG
    # entry and the old "was it explicitly set" membership check became useless.
    from my_project_orchestrator.agent import ProjectOrchestrator

    (tmp_path / ".git").mkdir()
    orch = ProjectOrchestrator()
    project = MagicMock()
    project.path = tmp_path
    project.config = {"orchestrator": {"parallel_mode": "auto"}}
    tasks = [MagicMock(), MagicMock()]

    with patch.object(
        ProjectOrchestrator, "_execute_parallel_worktrees", return_value=[]
    ) as wt:
        orch._execute_parallel(tasks, MagicMock(), project)
    wt.assert_called_once()


def test_parallel_mode_shared_does_not_use_worktrees_on_git_repo(tmp_path):
    from my_project_orchestrator.agent import ProjectOrchestrator

    (tmp_path / ".git").mkdir()
    orch = ProjectOrchestrator()
    project = MagicMock()
    project.path = tmp_path
    project.config = {"orchestrator": {"parallel_mode": "shared", "max_workers": 2}}
    tasks = []  # empty: shared path returns immediately, no real execution

    with patch.object(
        ProjectOrchestrator, "_execute_parallel_worktrees", return_value=[]
    ) as wt:
        orch._execute_parallel(tasks, MagicMock(), project)
    wt.assert_not_called()
