"""Build report generation, ported from /build Phase 6.

Produces a structured markdown report summarizing the build session.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from my_project_orchestrator.core.assessment import HealthCheck, ProjectAssessment
from my_project_orchestrator.core.models import Task
from my_project_orchestrator.core.scratchpad import Scratchpad
from my_project_orchestrator.core.modes import BuildMode
from my_project_orchestrator.core.validator import ValidationResult
from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)


class BuildReport:
    def __init__(
        self,
        mode: BuildMode,
        project_name: str,
        assessment: ProjectAssessment,
        start_time: datetime,
    ):
        self.mode = mode
        self.project_name = project_name
        self.assessment = assessment
        self.start_time = start_time
        self.end_time: Optional[datetime] = None
        self.completed_tasks: list[Task] = []
        self.failed_tasks: list[Task] = []
        self.deferred_tasks: list[Task] = []
        self.key_decisions: list[str] = []
        self.scratchpad: Optional[Scratchpad] = None
        self.health_before: Optional[HealthCheck] = None
        self.health_after: Optional[HealthCheck] = None
        self.validation: Optional[ValidationResult] = None
        self.validation_passed: Optional[bool] = None
        self.llm_calls: int = 0
        self.llm_tokens: int = 0
        self.llm_cost: float = 0.0
        self.llm_cache_read_tokens: int = 0
        self.cost_by_task: dict = {}
        # Best-effort subsystems that threw during the run (e.g. AB-MCTS,
        # probes). Surfaced in the report so a silently-dead subsystem is
        # visible to the morning reader, not just buried in a log.
        self.degraded_subsystems: list[str] = []
        # Unmet-goal gaps from the optional goal-completion check (advisory: the
        # gates can be green while the goal is not fully met). Empty unless the
        # check ran and returned a GAP verdict.
        self.goal_gaps: list[str] = []

    def finalize(self, end_time: Optional[datetime] = None):
        self.end_time = end_time or datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        """Structured representation for programmatic access / history."""
        return {
            "mode": self.mode.value,
            "project": self.project_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "completed": [t.id for t in self.completed_tasks],
            "failed": [t.id for t in self.failed_tasks],
            "deferred": [t.id for t in self.deferred_tasks],
            "validation_passed": self.validation_passed,
            "llm_calls": self.llm_calls,
            "llm_tokens": self.llm_tokens,
            "llm_cost": self.llm_cost,
            "degraded_subsystems": list(self.degraded_subsystems),
            "goal_gaps": list(self.goal_gaps),
        }

    def save(self, project_path: Path) -> Optional[Path]:
        """Persist the report (markdown + JSON) under .orchestrator/reports.

        Failure to write a report must never abort a build, so write errors are
        logged and swallowed rather than propagated.
        """
        try:
            reports_dir = Path(project_path) / ".orchestrator" / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            stamp = (self.end_time or self.start_time).strftime("%Y%m%d_%H%M%S")
            md_path = reports_dir / f"report_{stamp}.md"
            md_path.write_text(self.to_markdown(), encoding="utf-8")
            (reports_dir / f"report_{stamp}.json").write_text(
                json.dumps(self.to_dict(), indent=2), encoding="utf-8"
            )
            logger.info(f"Report saved to {md_path}")
            return md_path
        except OSError as e:
            logger.error(f"Failed to save build report: {e}")
            return None

    def _verdict_block(self) -> list[str]:
        """Top-line go/no-go verdict plus an evidence block, derived purely
        from existing report fields (no LLM calls, no I/O)."""
        n_completed = len(self.completed_tasks)
        n_failed = len(self.failed_tasks)
        n_deferred = len(self.deferred_tasks)
        nothing_done = (n_completed + n_failed + n_deferred) == 0

        if self.validation_passed is False or (
            nothing_done and self.validation_passed is not True
        ):
            verdict = "FAILED"
            reason = "build/test gate is red or nothing meaningful completed."
        elif self.validation_passed and n_failed == 0:
            verdict = "SHIP"
            reason = "validation passed and no tasks failed."
        else:
            verdict = "NEEDS REVIEW"
            reason = "build largely succeeded but has open issues; see below."

        lines = [f"## Verdict: {verdict}", f"_{reason}_\n", "### Evidence"]
        lines.append(
            f"- Tasks: {n_completed} completed, {n_failed} failed, {n_deferred} deferred"
        )
        if self.validation:
            lines.append(f"- Validation: {self.validation.summary()}")
        elif self.validation_passed is not None:
            lines.append(
                f"- Validation: {'passed' if self.validation_passed else 'failed'}"
            )
        if self.health_before or self.health_after:
            hb = self.health_before or HealthCheck()
            ha = self.health_after or HealthCheck()
            lines.append(
                f"- Health: builds {'YES' if hb.builds else 'NO'} -> "
                f"{'YES' if ha.builds else 'NO'}, "
                f"tests {hb.test_failures} fail -> {ha.test_failures} fail, "
                f"lint {hb.lint_warnings} -> {ha.lint_warnings} warnings"
            )
        if self.validation and self.validation.diff_stats:
            lines.append(f"- Diff: {self.validation.diff_stats}")
        if self.llm_calls > 0:
            lines.append(
                f"- LLM: {self.llm_calls} calls, {self.llm_tokens:,} tokens, "
                f"${self.llm_cost:.4f}"
            )

        blocking = list(self.failed_tasks)
        issues = list(self.validation.issues) if self.validation else []
        if blocking or issues:
            lines.append("\n**Blocking items:**")
            for t in blocking:
                lines.append(f"- Failed task {t.id}: {t.title or t.description[:60]}")
            for issue in issues:
                lines.append(f"- {issue}")
        if self.degraded_subsystems:
            names = ", ".join(d.split(":")[0] for d in self.degraded_subsystems)
            lines.append(f"\n**Degraded subsystems** (ran WITHOUT: {names}):")
            for d in self.degraded_subsystems:
                lines.append(f"- {d}")
        if self.goal_gaps:
            lines.append(
                "\n**Goal gaps** (advisory: gates passed but the goal may not be "
                "fully met):"
            )
            for gap in self.goal_gaps:
                lines.append(f"- {gap}")
        lines.append("")
        return lines

    def to_markdown(self) -> str:
        self.end_time = self.end_time or datetime.now(timezone.utc)
        duration = (self.end_time - self.start_time).total_seconds()
        duration_min = duration / 60

        s = self.assessment.structure
        langs = ", ".join(s.languages) if s.languages else "unknown"

        lines = [
            "## Build Report\n",
            f"**Mode**: {self.mode.value} | **Duration**: ~{duration_min:.1f} minutes",
            f"**Project**: {self.project_name} | **Type**: {s.project_type} | **Languages**: {langs}\n",
        ]

        # Go/no-go verdict: someone returning to an unattended run must see
        # "can I ship this?" up front, derived purely from existing fields,
        # before scanning the task tables below.
        lines.extend(self._verdict_block())

        # Validation banner: a failed quality gate must be visible at the top,
        # not buried while the report otherwise reads as a success.
        if self.validation_passed is not None:
            if self.validation_passed:
                lines.append("**Validation: PASSED**\n")
            else:
                lines.append(
                    "**Validation: FAILED** - quality gate did not pass; see issues below."
                )
                if self.validation and self.validation.issues:
                    for issue in self.validation.issues:
                        lines.append(f"- {issue}")
                lines.append("")

        # Health Before -> After
        if self.health_before or self.health_after:
            lines.append("### Health Before -> After")
            lines.append("| Check | Before | After |")
            lines.append("|-------|--------|-------|")
            hb = self.health_before or HealthCheck()
            ha = self.health_after or HealthCheck()
            lines.append(
                f"| Builds | {'YES' if hb.builds else 'NO'} | {'YES' if ha.builds else 'NO'} |"
            )
            lines.append(
                f"| Tests | {hb.test_count - hb.test_failures} pass, {hb.test_failures} fail | "
                f"{ha.test_count - ha.test_failures} pass, {ha.test_failures} fail |"
            )
            lines.append(
                f"| Lint | {hb.lint_warnings} warnings | {ha.lint_warnings} warnings |"
            )
            lines.append("")

        # Technical Debt & Risk
        lines.append("### Technical Debt & Risk")
        lines.append(f"**Debt Score**: {self.assessment.tech_debt.score}/100")
        lines.append(f"**Risk Level**: {self.assessment.risk.level.upper()}")
        if self.assessment.risk.factors:
            lines.append("\n**Risk Factors:**")
            for factor in self.assessment.risk.factors:
                lines.append(f"- {factor}")
        lines.append("")

        # Task summary
        lines.append("### Tasks")
        lines.append("| Status | Count |")
        lines.append("|--------|-------|")
        lines.append(f"| Completed | {len(self.completed_tasks)} |")
        lines.append(f"| Failed | {len(self.failed_tasks)} |")
        lines.append(f"| Deferred | {len(self.deferred_tasks)} |")
        lines.append("")

        # Completed tasks
        if self.completed_tasks:
            lines.append("### Completed Tasks")
            lines.append("| # | ID | Title |")
            lines.append("|---|------|-------|")
            for i, t in enumerate(self.completed_tasks, 1):
                lines.append(f"| {i} | {t.id} | {t.title or t.description[:60]} |")
            lines.append("")

        # Failed tasks
        if self.failed_tasks:
            lines.append("### Failed Tasks")
            lines.append("| ID | Title | Status |")
            lines.append("|------|-------|--------|")
            for t in self.failed_tasks:
                lines.append(
                    f"| {t.id} | {t.title or t.description[:60]} | {t.status} |"
                )
            lines.append("")

        # Deferred tasks
        if self.deferred_tasks:
            lines.append("### Deferred Tasks")
            lines.append("| ID | Title |")
            lines.append("|------|-------|")
            for t in self.deferred_tasks:
                lines.append(f"| {t.id} | {t.title or t.description[:60]} |")
            lines.append("")

        # Key decisions
        if self.key_decisions:
            lines.append("### Key Decisions")
            for d in self.key_decisions:
                lines.append(f"- {d}")
            lines.append("")

        # Scratchpad discoveries
        if self.scratchpad and len(self.scratchpad) > 0:
            lines.append("### Scratchpad Discoveries")
            for entry in self.scratchpad.entries:
                lines.append(f"- [{entry.category}] {entry.discovery}")
            lines.append("")

        # Validation
        if self.validation:
            lines.append("### Validation")
            lines.append(f"- {self.validation.summary()}")
            if self.validation.diff_stats:
                lines.append(f"\n```\n{self.validation.diff_stats}\n```")
            lines.append("")

        # LLM usage
        if self.llm_calls > 0:
            lines.append("### LLM Usage")
            lines.append(
                f"- {self.llm_calls} calls, {self.llm_tokens:,} tokens, "
                f"${self.llm_cost:.4f} estimated cost"
            )
            if self.llm_cache_read_tokens > 0 and self.llm_tokens > 0:
                rate = 100.0 * self.llm_cache_read_tokens / self.llm_tokens
                lines.append(
                    f"- Cache: {self.llm_cache_read_tokens:,} tokens read from cache ({rate:.0f}% of total)"
                )
            if self.cost_by_task:
                top = sorted(
                    self.cost_by_task.items(), key=lambda kv: kv[1], reverse=True
                )[:5]
                lines.append("- Most expensive tasks:")
                for tid, cost in top:
                    lines.append(f"  - {tid}: ${cost:.4f}")
            lines.append("")

        # Summary line
        total = (
            len(self.completed_tasks)
            + len(self.failed_tasks)
            + len(self.deferred_tasks)
        )
        validation_note = ""
        if self.validation_passed is False:
            validation_note = " VALIDATION FAILED."
        lines.append("---")
        lines.append(
            f"*{len(self.completed_tasks)}/{total} tasks completed.{validation_note} "
            f"Duration: {duration_min:.1f}m.*"
        )

        return "\n".join(lines)
