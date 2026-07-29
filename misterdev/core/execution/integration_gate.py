"""Post-wave integration gate + regression revert, extracted from the orchestrator.

``ProjectOrchestrator`` mixes this in: every method uses only ``self`` and the
``executor`` passed to it, so the split is behavior-preserving. The unit owns the
"did merging this wave regress the suite, and if so revert the culprits" concern:
the full-suite gate (binary / count / identity modes for a red baseline), the
per-target polyglot analogue, the shared failure-count / failing-id parsers, and
the post-build bisect-and-revert.
"""

from typing import Optional

from misterdev.core.execution.project import Project
from misterdev.core.models import Task
from misterdev.core.modes import BuildFlags
from misterdev.core.planning.assessment import ProjectAssessment
from misterdev.core.reporting.report import BuildReport
from misterdev.logging_setup import setup_logger
from misterdev.task_executors.markdown_plan_executor import MarkdownPlanExecutor

logger = setup_logger(__name__)


class IntegrationGateMixin:
    @staticmethod
    def _wave_commits(executor, project, tasks) -> list:
        """Collect ``(task_id, sha)`` for each task that has a recorded commit,
        skipping tasks with none. Shared by the regression-revert and
        integration-gate paths."""
        commits = []
        for t in tasks:
            sha = executor.find_task_commit(project, t.id)
            if sha:
                commits.append((t.id, sha))
        return commits

    def _maybe_rollback_regression(
        self,
        project: Project,
        report: BuildReport,
        assessment: ProjectAssessment,
        flags: BuildFlags,
    ) -> None:
        """If the post-build gate failed, bisect task commits and revert the culprit."""
        if flags.no_rollback or not report.completed_tasks:
            return
        test_cmd = assessment.structure.test_command
        if not test_cmd:
            return
        ex = MarkdownPlanExecutor()
        if not ex._is_git_repo(project):
            return
        commits = self._wave_commits(ex, project, report.completed_tasks)
        if not commits:
            return
        logger.warning("Post-build regression detected; bisecting task commits...")
        culprit = ex.bisect_regression(project, commits, test_cmd)
        if not culprit:
            logger.info("Bisect did not isolate a single task commit.")
            return
        sha = dict(commits)[culprit]
        if ex.revert_task_commit(project, sha):
            logger.warning(
                f"Regression bisected to {culprit}; commit {sha[:8]} reverted."
            )
            report.key_decisions.append(
                f"Regression from {culprit} auto-reverted (bisect)"
            )

    def _suite_failures(
        self,
        project: Project,
        executor: MarkdownPlanExecutor,
        test_cmd: str,
        timeout: int,
        cwd=None,
    ) -> Optional[int]:
        """Full-suite failure count: 0 when green, the parsed count when red, or
        None when the count can't be parsed (caller then can't count-compare).

        ``cwd`` runs the command in a sub-project (target) directory; defaults to
        the repo root."""
        from misterdev.core.verification.validator import (
            _parse_test_counts,
        )

        ok, output = executor._run_command(project, test_cmd, timeout=timeout, cwd=cwd)
        if ok:
            return 0
        total, failures = _parse_test_counts(output)
        return failures if total > 0 else None

    @staticmethod
    def _failing_ids_from_output(output: str, project: Project) -> Optional[set]:
        """The SET of failing test identifiers parsed from runner output, or None
        when none can be parsed (caller falls back to the count).

        Identity beats a bare count: it lets the integration gate revert a wave
        that offsets a genuine fix against a NEW break (net-zero count, which
        count mode waves through) and stays correct if a fix renames/reorders
        tests. Reuses the FailureView parsers already validated per runner."""
        from misterdev.core.execution.failure_view import extract_failures

        lang = (
            (project.config.get("language") or "")
            if getattr(project, "config", None)
            else ""
        )
        ids = {
            f.test
            for f in extract_failures(output, language=lang)
            if getattr(f, "test", "")
        }
        return ids or None

    def _suite_failing_ids(
        self,
        project: Project,
        executor: MarkdownPlanExecutor,
        test_cmd: str,
        timeout: int,
        cwd=None,
    ) -> Optional[set]:
        """Full-suite failing-test id SET: empty when green, the parsed ids when
        red, or None when unparseable (caller falls back to the count)."""
        ok, output = executor._run_command(project, test_cmd, timeout=timeout, cwd=cwd)
        if ok:
            return set()
        return self._failing_ids_from_output(output, project)

    @staticmethod
    def _looks_like_broken_build(output: str) -> bool:
        """True when a FAILING suite's output shows a structural break — the code no
        longer compiles, collects, or imports — rather than ordinary (countable)
        test failures we merely could not parse. A build break is strictly worse
        than any counted red baseline (the suite no longer even runs), so it must be
        reverted even though it yields no count; an unrecognized-but-ran failure
        stays ambiguous and is left alone."""
        from misterdev.core.execution.compile_view import extract_compile_errors

        if extract_compile_errors(output):
            return True
        low = (output or "").lower()
        signals = (
            "error collecting",
            "errors during collection",
            "importerror",
            "modulenotfounderror",
            "no module named",
            "syntaxerror",
            "internalerror",
            "cannot find module",
            "cannot import",
            "unresolved import",
            "segmentation fault",
        )
        return any(s in low for s in signals)

    def _suite_broken(
        self, project, executor, test_cmd: str, timeout: int, cwd=None
    ) -> bool:
        """True when the suite command fails AND its output is a build/collection
        break. Runs the command once; used only on the unparseable path to tell a
        genuine break (revert) from an ambiguous unrecognized failure (leave)."""
        ok, output = executor._run_command(project, test_cmd, timeout=timeout, cwd=cwd)
        return (not ok) and self._looks_like_broken_build(output)

    def _integration_gate_count(
        self,
        project: Project,
        executor: MarkdownPlanExecutor,
        test_cmd: str,
        wave_tasks: list[Task],
        timeout: int,
        baseline_failures: int,
    ) -> list[str]:
        """Count-mode gate for a RED baseline: revert wave commits (newest first)
        only when the wave RAISED the full-suite failure count above the baseline.

        This closes the gap where, with the binary gate disabled by a red
        baseline, a task gated on its own scoped tests could worsen the overall
        suite and still commit. An unparseable post-wave count is left alone (we
        do not revert on a number we can't read).
        """
        ok, raw_output = executor._run_command(project, test_cmd, timeout=timeout)
        if ok:
            after: Optional[int] = 0
        else:
            from misterdev.core.verification.validator import _parse_test_counts

            total, failures = _parse_test_counts(raw_output)
            after = failures if total > 0 else None
        if after is None:
            if not self._looks_like_broken_build(raw_output):
                return []
            logger.warning(
                "Integration gate (count): post-wave suite no longer builds/collects "
                "(strictly worse than the red baseline); reverting wave commits."
            )
        elif after <= baseline_failures:
            return []
        else:
            logger.warning(
                f"Integration gate (count): failures rose {baseline_failures} -> "
                f"{after}; reverting wave commits until restored."
            )
        commits = self._wave_commits(executor, project, wave_tasks)
        reverted: list[str] = []
        for tid, sha in reversed(commits):
            if executor.revert_task_commit(project, sha):
                reverted.append(tid)
            now = self._suite_failures(project, executor, test_cmd, timeout)
            if now is not None and now <= baseline_failures:
                break
        return reverted

    def _integration_gate_ids(
        self,
        project: Project,
        executor: MarkdownPlanExecutor,
        test_cmd: str,
        wave_tasks: list[Task],
        timeout: int,
        baseline_ids: set,
    ) -> list[str]:
        """Identity-mode gate for a RED baseline: revert the wave iff it introduced
        a NEW failing test (one not failing at baseline), regardless of the failure
        COUNT.

        Stricter and more correct than count mode: a wave that fixes test A but
        breaks test B nets zero on the count and slips past ``_integration_gate_count``,
        yet it introduced a real regression (B) — identity mode reverts it. A wave
        that resolves none of the baseline failures and adds none (a no-op "fix"
        that still merged) is surfaced as no-progress rather than silently blessed.
        """
        after = self._suite_failing_ids(project, executor, test_cmd, timeout)
        if after is None:
            if not self._suite_broken(project, executor, test_cmd, timeout):
                return []  # unparseable-but-ran; don't revert on ids we can't read
            logger.warning(
                "Integration gate (identity): post-wave suite no longer builds/"
                "collects; reverting wave commits until restored."
            )
            commits = self._wave_commits(executor, project, wave_tasks)
            reverted: list[str] = []
            for tid, sha in reversed(commits):
                if executor.revert_task_commit(project, sha):
                    reverted.append(tid)
                now = self._suite_failing_ids(project, executor, test_cmd, timeout)
                if now is not None and not (now - baseline_ids):
                    break
            return reverted
        new_failures = after - baseline_ids
        if not new_failures:
            if baseline_ids - after:
                logger.info(
                    "Integration gate (identity): resolved "
                    f"{len(baseline_ids - after)} baseline failure(s), no regressions."
                )
            else:
                logger.info(
                    "Integration gate (identity): wave added no new failures but "
                    "resolved none either — no progress on the failing suite."
                )
            return []
        logger.warning(
            f"Integration gate (identity): {len(new_failures)} new failing test(s) "
            f"(e.g. {sorted(new_failures)[:2]}); reverting wave commits until restored."
        )
        commits = self._wave_commits(executor, project, wave_tasks)
        reverted: list[str] = []
        for tid, sha in reversed(commits):
            if executor.revert_task_commit(project, sha):
                reverted.append(tid)
            now = self._suite_failing_ids(project, executor, test_cmd, timeout)
            if now is not None and not (now - baseline_ids):
                break
        return reverted

    @staticmethod
    def _target_regressed(after: Optional[int], baseline: Optional[int]) -> bool:
        """Did a target's gate regress vs its baseline?

        ``after``/``baseline`` are :meth:`_suite_failures` results (0 green, N
        count, None unparseable). A green-now gate never regressed. With no
        countable baseline we can't compare, so we don't revert. A binary failure
        now (None) is a regression only if the target was green (baseline 0);
        otherwise compare counts.
        """
        if after == 0:
            return False
        if baseline is None:
            return False
        if after is None:
            return baseline == 0
        return after > baseline

    def _integration_gate_targets(
        self,
        project: Project,
        executor: MarkdownPlanExecutor,
        targets: list[dict],
        wave_tasks: list[Task],
        timeout: int,
        target_baselines: dict,
    ) -> list[str]:
        """Per-target integration gate: validate each sub-project the wave touched
        with ITS own toolchain (in ITS directory), reverting only the wave commits
        belonging to a target that regressed. This is the multi-target analogue of
        :meth:`_integration_gate` — the last place a polyglot run would otherwise
        gate with the wrong toolchain.
        """
        from misterdev.core.planning.targets import select_target

        reverted: list[str] = []
        for tgt in targets:
            gate_cmd = tgt.get("test_command") or tgt.get("build_command")
            if not gate_cmd:
                continue
            tname = tgt.get("name") or tgt.get("path")
            tp = (tgt.get("path") or "").strip("/")
            run_dir = project.path / tp if tp else project.path
            owned = [
                t
                for t in wave_tasks
                if (
                    select_target(
                        targets, list(t.files_to_modify) + list(t.files_to_create)
                    )
                    or {}
                ).get("path")
                == tgt.get("path")
            ]
            if not owned:
                continue
            baseline = target_baselines.get(tname)
            after = self._suite_failures(
                project, executor, gate_cmd, timeout, cwd=run_dir
            )
            if not self._target_regressed(after, baseline):
                continue
            logger.warning(
                f"Integration gate [{tname}]: regressed (baseline={baseline}, "
                f"after={after}); reverting this target's wave commits."
            )
            commits = [(t.id, executor.find_task_commit(project, t.id)) for t in owned]
            commits = [(tid, sha) for tid, sha in commits if sha]
            for tid, sha in reversed(commits):
                if executor.revert_task_commit(project, sha):
                    reverted.append(tid)
                now = self._suite_failures(
                    project, executor, gate_cmd, timeout, cwd=run_dir
                )
                if not self._target_regressed(now, baseline):
                    break
        return reverted

    def _integration_gate(
        self,
        project: Project,
        executor: MarkdownPlanExecutor,
        test_cmd: str,
        wave_tasks: list[Task],
        timeout: int,
        baseline_failures: int = 0,
    ) -> list[str]:
        """Run the full suite after a wave; revert task commits that regressed it.

        Returns the task_ids whose commits were reverted (empty if the suite
        still passes). Bisects to the single culprit when possible; if that
        can't isolate it or the tree is still red afterward, reverts the
        remaining wave commits (newest first) to restore a green baseline. On a
        RED baseline it prefers IDENTITY mode (revert a wave that adds any new
        failing test, so an offsetting fix/break can't slip through) when the
        baseline's failing set was parseable, falling back to COUNT mode (revert
        only a wave that raises the failure count) otherwise.
        """
        # Prefer identity mode (revert on any NEW failing test) over count mode
        # (revert only when the count rises) whenever the baseline's failing set
        # was parseable: it also catches an offsetting fix/break that count mode
        # nets to zero. Count mode remains the fallback for unparseable output.
        baseline_ids = getattr(project, "baseline_test_failing_ids", None)
        if baseline_ids:
            return self._integration_gate_ids(
                project, executor, test_cmd, wave_tasks, timeout, baseline_ids
            )
        if baseline_failures > 0:
            return self._integration_gate_count(
                project, executor, test_cmd, wave_tasks, timeout, baseline_failures
            )
        ok, _ = executor._run_command(project, test_cmd, timeout=timeout)
        if ok:
            return []

        commits = self._wave_commits(executor, project, wave_tasks)
        if not commits:
            logger.warning(
                "Integration gate: suite regressed but no task commits found to revert."
            )
            return []

        logger.warning("Integration gate: suite regressed; isolating culprit...")
        reverted: list[str] = []
        culprit = executor.bisect_regression(
            project, commits, test_cmd, timeout=timeout
        )
        if culprit:
            sha = dict(commits)[culprit]
            if executor.revert_task_commit(project, sha):
                reverted.append(culprit)
                ok, _ = executor._run_command(project, test_cmd, timeout=timeout)
                if ok:
                    return reverted

        for tid, sha in reversed(commits):
            if tid in reverted:
                continue
            if executor.revert_task_commit(project, sha):
                reverted.append(tid)
            ok, _ = executor._run_command(project, test_cmd, timeout=timeout)
            if ok:
                break
        return reverted
