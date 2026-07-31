"""ReportingMixin — post-run learning persistence and run-summary emission.

Extracted from agent.py. _persist_learning calls _record_env_learnings and
_write_run_summary; _write_run_summary calls _task_failure_text and
_emit_run_summary. All five are kept together to avoid cross-mixin calls.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rich.console import Console

from misterdev.agent_helpers import (
    worktree_healthcheck_command,
    worktree_setup_command,
)
from misterdev.config import get_setting
from misterdev.core.execution.env_learnings import EnvLearnings
from misterdev.core.execution.project import Project
from misterdev.core.learning import FailureLog, SolvedTaskIndex
from misterdev.core.reporting.report import BuildReport
from misterdev.logging_setup import setup_logger
from misterdev.utils.file_utils import atomic_write, orchestrator_state_file

logger = setup_logger(__name__)
_console = Console()


class ReportingMixin:
    def _persist_learning(self, project: Project, report: BuildReport) -> None:
        """Record this build's spend, real failures, and solved tasks.

        Runs on EVERY terminated build — normal completion and budget halt alike.
        A budget-exhausted run spent the whole cap and still failed, so it is the
        highest-signal failure; dropping it would blind the exact features
        (evolution-from-failures, warm-start) that learn from real use. Each write
        is best-effort so bookkeeping never turns a finished build into a crash.
        """
        # Expose this build's spend so a caller (e.g. the benchmark runner) can
        # attribute per-run cost without re-deriving it from the saved report.
        self.last_build_cost = float(
            getattr(project.llm_client.cumulative_usage, "estimated_cost", 0.0)
        )
        try:
            FailureLog(
                project.path / ".orchestrator" / "failures.jsonl"
            ).record_failures(report.failed_tasks)
        except Exception as e:
            logger.warning(f"Failure logging failed (non-fatal): {e}")
            report.degraded_subsystems.append(f"Failure logging: {e}")
        try:
            SolvedTaskIndex(
                project.path / ".orchestrator" / "solved_tasks.jsonl"
            ).record(report.completed_tasks)
        except Exception as e:
            logger.warning(f"Solved-task indexing failed (non-fatal): {e}")
            report.degraded_subsystems.append(f"Solved-task indexing: {e}")
        try:
            self._record_env_learnings(project)
        except Exception as e:
            logger.warning(f"Env-memory persist failed (non-fatal): {e}")
            report.degraded_subsystems.append(f"Env-memory persist: {e}")
        try:
            self._write_run_summary(project, report)
        except Exception as e:
            logger.warning(f"Run summary write failed (non-fatal): {e}")
            report.degraded_subsystems.append(f"Run summary: {e}")
        try:
            project.model_ledger.merge_into(
                Path.home() / ".misterdev" / "model_stats.json"
            )
        except Exception as e:
            logger.warning(f"Global model ledger merge failed (non-fatal): {e}")

    @staticmethod
    def _task_failure_text(task) -> str:
        """The best-available failure text for a terminal non-success task: the
        error stashed on the error path, else the last execution result's message
        and logs. '' when nothing is recorded."""
        stored = (getattr(task, "processor_data", None) or {}).get("failure_text")
        if stored:
            return str(stored)
        hist = getattr(task, "execution_history", None) or []
        if hist:
            last = hist[-1]
            return f"{getattr(last, 'message', '') or ''}\n{getattr(last, 'logs', '') or ''}"
        return ""

    def _write_run_summary(self, project: Project, report: BuildReport) -> None:
        """Classify the build pipeline's failures and write the one-glance summary
        (feeds P7/P10; answers "why did this run underperform" at a glance)."""
        end = report.end_time or datetime.now(timezone.utc)
        elapsed = (end - report.start_time).total_seconds()
        failed_items = [(t.id, self._task_failure_text(t)) for t in report.failed_tasks]
        deferred_items = [
            (t.id, self._task_failure_text(t)) for t in report.deferred_tasks
        ]
        self._emit_run_summary(
            project, len(report.completed_tasks), failed_items, deferred_items, elapsed
        )

    def _emit_run_summary(
        self,
        project: Project,
        completed: int,
        failed_items: list,
        deferred_items: list,
        elapsed_seconds: float,
    ) -> None:
        """Classify (id, text) failure/deferral pairs into the taxonomy and write a
        one-glance summary to the console and ``.orchestrator/run_summary.json``.
        Shared by the build pipeline and the ``run --tasks`` path so both emit the
        same signal. Never raises — a summary must not sink a finished run."""
        from misterdev.core.execution.failure_taxonomy import build_run_summary

        summary = build_run_summary(
            completed, failed_items, deferred_items, elapsed_seconds
        )
        atomic_write(
            orchestrator_state_file(project.path, "run_summary.json"),
            json.dumps(summary, indent=2),
        )
        mins, secs = divmod(int(summary["elapsed_seconds"]), 60)
        parts = [f"[green]{summary['completed']} done[/]"]
        if summary["deferred"]:
            parts.append(f"[yellow]{summary['deferred']} deferred[/]")
        if summary["failed"]:
            parts.append(f"[red]{summary['failed']} failed[/]")
        _console.print(
            "[bold]Run summary[/] · "
            + " · ".join(parts)
            + f" · [dim]{mins}m {secs}s[/]"
        )
        if summary["failure_breakdown"]:
            brk = ", ".join(
                f"{cat} {n}" for cat, n in summary["failure_breakdown"].items()
            )
            _console.print(f"  [dim]failures:[/] {brk}")
            top = summary["top_obstacle"]
            if top:
                ex = summary["exemplars"].get(top, "")
                _console.print(
                    f"  [dim]top obstacle:[/] {top}"
                    + (f" [dim]— {ex[:120]}[/]" if ex else "")
                )

    def _record_env_learnings(self, project: Project) -> None:
        """Record this run's durable environment facts for the next run.

        Loads the existing ledger and refreshes: the effective worktree
        setup/healthcheck commands (resolved from the current config), and a
        learned max_workers — persisted ONLY when the adaptive loop settled BELOW
        the configured base (a real, contention-driven reduction). A run that held
        or recovered to full concurrency clears any stale reduction, so a single
        bad run never pins the project low forever.
        """
        learnings = EnvLearnings.load(project.path)
        setup = worktree_setup_command(project.config, project.path)
        if setup:
            learnings.worktree_setup_command = setup
        health = worktree_healthcheck_command(project.config, project.path)
        if health:
            learnings.worktree_healthcheck_command = health
        settled = getattr(project, "env_settled_workers", None)
        base = getattr(project, "env_base_workers", None)
        if settled is not None and base is not None:
            learnings.max_workers = settled if settled < base else None
        learnings.save(project.path)
