"""GoalCheckMixin — goal-completion verification and learning-embedder setup.

Extracted from agent.py. _run_goal_check calls _cumulative_diff; all four
methods are pure functions of their arguments with no back-refs to other self
methods.
"""

from typing import Optional

from misterdev.config import get_setting
from misterdev.core.execution.project import Project
from misterdev.core.gitcmd import run_git
from misterdev.core.reporting.report import BuildReport
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)


class GoalCheckMixin:
    def _learning_embedder(self, project: Project):
        """A shared embedder for lesson/warm-start retrieval, or None.

        Reuses the project's embedding-backend config (which prefers a free,
        offline local model and honours "none"), so semantic ranking is opt-in via
        the same setting that governs context ranking. Any failure returns None,
        degrading retrieval to lexical rather than breaking the build."""
        try:
            from misterdev.llm.client.embeddings import create_embedding_client

            return create_embedding_client(project.config)
        except Exception as e:
            logger.debug(f"Learning embedder unavailable, using lexical ranking: {e}")
            return None

    def _capture_head(self, project: Project) -> Optional[str]:
        """Best-effort current HEAD sha, or None outside a git repo / on error."""
        if not (project.path / ".git").exists():
            return None
        proc = run_git("git rev-parse HEAD", project.path)
        if proc is None:
            return None
        sha = proc.stdout.strip()
        return sha if proc.returncode == 0 and sha else None

    def _cumulative_diff(self, project: Project, base: Optional[str]) -> str:
        """Diff of the whole build's work for the goal-check judge.

        When a pre-build base sha is known, diff ``base`` against the working
        tree (committed task commits + uncommitted changes). Otherwise fall back
        to the working-tree diff vs HEAD. Best-effort: returns "" on any error or
        outside a git repo, which the judge treats as no diff.
        """
        if not (project.path / ".git").exists():
            return ""
        cmd = f"git diff {base}" if base else "git diff HEAD"
        proc = run_git(cmd, project.path)
        return proc.stdout if proc and proc.returncode == 0 else ""

    def _run_goal_check(
        self,
        project: Project,
        prompt: str,
        tasks: list,
        base: Optional[str],
        report: BuildReport,
    ) -> None:
        """Run the optional goal-completion check and record its verdict.

        Advisory by default: a GAP verdict records gaps into the report and logs
        them but does NOT fail the build. It blocks (marks validation failed and
        appends a blocking issue) only when ``orchestrator.block_on_goal_gap`` is
        true. SKIP (no goal/criteria/client, unparseable, timeout, error) is a
        no-op. Wrapped so a judge failure can never crash a finished build.
        """
        from misterdev.core.verification.goal_check import (
            GAP,
            build_evidence,
            run_goal_check,
        )

        try:
            criteria = "\n".join(
                f"- {t.acceptance_criteria}"
                for t in tasks
                if getattr(t, "acceptance_criteria", "")
            )
            diff = self._cumulative_diff(project, base)
            summary = "; ".join(
                t.title or t.description[:60] for t in report.completed_tasks
            )
            evidence = build_evidence(diff=diff, summary=summary)
            timeout = get_setting(project.config, "orchestrator", "goal_check_timeout")
            judge_model = (project.config.get("judge") or {}).get("model")
            verdict = run_goal_check(
                prompt,
                criteria,
                evidence,
                llm_client=project.llm_client,
                judge_model=judge_model,
                timeout=timeout,
            )
        except Exception as e:
            logger.warning(f"Goal-completion check failed (non-fatal): {e}")
            report.degraded_subsystems.append(f"Goal-completion check: {e}")
            return

        if verdict.status != GAP:
            logger.info(f"Goal-completion check: {verdict.status} ({verdict.reason})")
            return

        report.goal_gaps = list(verdict.gaps)
        logger.warning(f"Goal-completion check found {len(verdict.gaps)} gap(s):")
        for gap in verdict.gaps:
            logger.warning(f"  goal gap: {gap}")

        if get_setting(project.config, "orchestrator", "block_on_goal_gap"):
            report.validation_passed = False
            self.last_build_succeeded = False
            issue = "Goal-completion check: " + "; ".join(verdict.gaps)
            if report.validation is not None:
                report.validation.issues.append(issue)
            else:
                report.key_decisions.append(issue)
