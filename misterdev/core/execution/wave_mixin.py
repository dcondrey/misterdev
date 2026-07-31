"""WaveMixin — per-wave helper utilities for ProjectOrchestrator.

Extracted from agent.py. All three methods are pure functions of their
arguments; none reference other self methods.
"""

from misterdev.config import get_setting
from misterdev.core.execution.project import Project
from misterdev.core.planning.assessment import HealthCheck
from misterdev.core.reporting.report import BuildReport
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)


class WaveMixin:
    def _build_fix_spec(
        self,
        report: BuildReport,
        issues: list[str],
        final_health: HealthCheck,
    ) -> str:
        """Compose a targeted spec from the gate's concrete failures.

        Used by the convergence loop for iterations 2+: instead of re-running
        expensive discovery, it points decomposition straight at what the gate
        flagged (build/test/lint output, failed and deferred tasks) so the next
        pass fixes the gap rather than re-planning the whole build.
        """
        parts = ["# Convergence Fix Spec", "## Goal: make the gate pass\n"]
        if issues:
            parts.append("### Gate Failures")
            for item in issues:
                parts.append(f"- {item}")
        if not final_health.builds and final_health.build_output:
            parts.append(f"\n### Build Output\n{final_health.build_output[:1000]}")
        if not final_health.tests_pass and final_health.test_output:
            parts.append(f"\n### Test Output\n{final_health.test_output[:1000]}")
        if not final_health.lint_clean and final_health.lint_output:
            parts.append(f"\n### Lint Output\n{final_health.lint_output[:1000]}")
        if report.failed_tasks:
            parts.append("\n### Failed Tasks")
            for t in report.failed_tasks:
                parts.append(f"- {t.id}: {t.title}")
        if report.deferred_tasks:
            parts.append("\n### Deferred Tasks")
            for t in report.deferred_tasks:
                parts.append(f"- {t.id}: {t.title}")
        return "\n".join(parts)

    @staticmethod
    def _wave_infra_count(results: list) -> int:
        """How many of a wave's tasks FAILED on an ENVIRONMENT fault (not code).

        Scans each unsuccessful task's error/logs for an infra signature (timeout,
        locked store, OOM, ...). A completed task never counts — a transient fault
        it self-healed past is not contention worth backing off for. Only an
        UN-recovered infra failure, the exact signal that concurrency is too high,
        is counted.
        """
        from misterdev.core.execution.infra import infra_failure

        count = 0
        for _task, result, error in results:
            if result is not None and getattr(result, "status", None) == "completed":
                continue
            text = ""
            if error is not None:
                text += str(error)
            if result is not None:
                text += " " + str(getattr(result, "logs", "") or "")
                text += " " + str(getattr(result, "message", "") or "")
            if infra_failure(text):
                count += 1
        return count

    def _apply_wave_tuning(self, project: Project, tuning, base: dict) -> None:
        """Apply a wave's tuning by scaling the config the deep gate paths read.

        max_workers and the gate/setup timeouts are resolved via ``get_setting``
        throughout the executor and worktree code, so the one central way to make
        an adapted value reach all of them is to set it on the config for the
        wave. Always computed from the captured ``base`` (never the last wave's
        already-scaled value) so repeated application cannot drift. Safe between
        waves: the wave loop is serial here, and each wave's workers read the value
        once before the parallel section starts.
        """
        orch = project.config.setdefault("orchestrator", {})
        orch["max_workers"] = tuning.max_workers
        orch["worktree_setup_timeout"] = int(
            round(base["setup"] * tuning.timeout_factor)
        )
        build = project.config.setdefault("build", {})
        build["build_timeout"] = int(round(base["build"] * tuning.timeout_factor))
        build["test_timeout"] = int(round(base["test"] * tuning.timeout_factor))
