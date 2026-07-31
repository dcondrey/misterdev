"""PipelineMixin — build pipeline phases 1.5-6 and their support helpers.

Extracted from agent.py. _run_pipeline is the convergence loop (probes →
spec → decompose → execute → gate → repeat); the other six methods are its
collaborators, kept here to avoid cross-mixin calls.
"""

import threading
from pathlib import Path
from typing import Callable, Optional

from rich.console import Console
from rich.markdown import Markdown

from misterdev.agent_helpers import (
    _budget_exhausted,
    _check_golden_config,
)
from misterdev.analyzers.project_analyzer.detection import (
    detect_lint_command,
    detect_typecheck_command,
)
from misterdev.config import get_setting
from misterdev.core.execution.env_learnings import EnvLearnings
from misterdev.core.execution.progress import ProgressTracker, compute_task_hash
from misterdev.core.execution.project import Project
from misterdev.core.learning import SolvedTaskIndex
from misterdev.core.modes import BuildFlags, BuildMode
from misterdev.core.models import Task
from misterdev.core.planning.assessment import ProjectAssessment
from misterdev.core.planning.decomposer import (
    _split_oversized_tasks,
    decompose_spec,
    format_plan,
    split_keystone_tasks,
    topological_sort,
)
from misterdev.core.planning.metacognition import SessionAuditor
from misterdev.core.planning.sovereign import (
    ABMCTSPlanner,
    EphemeralCodeManager,
    ProbeGenerator,
)
from misterdev.core.reporting.report import BuildReport
from misterdev.core.verification.gatekeeper import GateKeeper
from misterdev.core.verification.validator import ValidationResult
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)
_console = Console()

CONVERGENCE_CEILING = 25


class PipelineMixin:
    def _requirements_preflight(self, project, tasks, proceed: bool) -> bool:
        """Review the plan for user-supplied inputs up front. Writes REQUIREMENTS.md,
        injects any typed decision answers, prints a summary, and — the smart gate —
        returns False (stop) only when a MISSING, cascade-wide input needs the user
        first. ``--proceed`` (or ``gather_requirements: false``) always returns True.
        Best-effort: any failure degrades to "proceed"."""
        if proceed or not get_setting(
            project.config, "orchestrator", "gather_requirements"
        ):
            return True
        try:
            from misterdev.core.planning.requirements import (
                RequirementsBook,
                gating_requirements,
                review_requirements,
            )

            llm = None
            if get_setting(project.config, "orchestrator", "requirements_llm_review"):
                llm = lambda p, s: project.llm_client.generate_code(p, s)  # noqa: E731
            reqs = review_requirements(tasks, llm=llm)
            if not reqs:
                return True

            book = RequirementsBook(project.path / ".orchestrator")
            book.write(reqs)
            for key, answer in book.load_answers().items():
                for r in reqs:
                    if r["key"] == key:
                        r["answered"] = True
                        for t in tasks:
                            if t.id in r.get("task_ids", []):
                                t.processor_data["user_answer"] = answer

            missing = [r for r in reqs if not r.get("satisfied")]
            _console.print(
                f"\n[bold]Requirements review:[/] {len(reqs)} input(s), "
                f"{len(missing)} not yet provided (see "
                f"{project.path / '.orchestrator' / 'REQUIREMENTS.md'})."
            )
            for r in missing[:10]:
                _console.print(f"  [yellow]✗[/] {r['key']} — {r.get('summary', '')}")

            gating = gating_requirements(reqs, tasks)
            if gating:
                keys = ", ".join(g["key"] for g in gating)
                _console.print(
                    f"[red]Stopping before execution:[/] {keys} are required by "
                    "foundational tasks and would cascade. Provide them (see "
                    "REQUIREMENTS.md), then re-run — or pass [bold]--proceed[/] to "
                    "run now and park what's missing."
                )
                return False
            if missing:
                _console.print(
                    "[dim]All missing inputs are late/leaf — proceeding; they will "
                    "park at the end if still unprovided.[/]"
                )
            return True
        except Exception as e:  # a preflight review must never block a run itself
            logger.warning(f"Requirements preflight skipped ({e}).")
            return True

    def _inject_task_context(
        self, task, contracts, changes, strategy_optimizer, project
    ) -> None:
        """Populate a task's processor_data with contracts, recent changes, and strategy."""
        task.processor_data["interface_contracts"] = contracts.get_contracts_for_task(
            task.dependencies
        )
        task.processor_data["recent_changes"] = changes.get_recent_changes_for_files(
            task.files_to_modify + task.files_to_create
        )
        task.processor_data["strategy"] = strategy_optimizer.select_best_strategy(
            task.description, task.category, "", project.llm_client
        )

    def _print_execution_plan(self, tasks: list[Task]) -> None:
        """Print the dependency-ordered wave plan without executing anything."""
        completed: set[str] = set()
        remaining = list(tasks)
        wave = 0
        _console.print(f"\n[bold]Execution Plan (dry-run): {len(tasks)} tasks[/]")
        while remaining:
            ready = [
                t for t in remaining if all(d in completed for d in t.dependencies)
            ]
            if not ready:
                _console.print(
                    f"[red]Dependency deadlock among: {[t.id for t in remaining]}[/]"
                )
                break
            wave += 1
            _console.print(f"\n[bold cyan]Wave {wave}[/] ({len(ready)} parallel):")
            for t in ready:
                deps = f" -> depends on {t.dependencies}" if t.dependencies else ""
                _console.print(
                    f"  [{t.id}] {t.title or t.description[:50]} ({t.complexity}, {t.category}){deps}"
                )
                completed.add(t.id)
            remaining = [t for t in remaining if t.id not in completed]
        _console.print(f"\n[dim]Total: {len(tasks)} tasks, {wave} waves.[/]\n")

    def _print_rerun_status(
        self, tasks: list[Task], progress, project_path, force: bool
    ) -> None:
        """Show which tasks would run vs skip, based on content hashes."""
        _console.print(f"\n[bold]Task status: {len(tasks)} tasks[/]")
        run = 0
        for t in tasks:
            rerun = force or progress.needs_rerun(
                t.id, compute_task_hash(t, project_path)
            )
            if rerun:
                run += 1
                reason = (
                    "forced"
                    if force
                    else ("changed" if progress.is_done(t.id) else "pending")
                )
                _console.print(f"  [yellow]RUN [/] {t.id}  ({reason})")
            else:
                _console.print(f"  [green]SKIP[/] {t.id}  (unchanged, completed)")
        _console.print(f"\n[dim]{run} would run, {len(tasks) - run} would skip.[/]\n")

    def _halt_on_budget(
        self, project: Project, report: Optional[BuildReport], error: Exception
    ) -> str:
        """Degrade a budget-exhausted run to a partial report (never a traceback).

        Records the halt, finalizes whatever work completed, and returns the
        report markdown. When the cap is hit before the report exists (during
        analysis), returns a concise message instead.
        """
        self.last_build_succeeded = False
        logger.error(f"Build halted by budget ceiling: {error}")
        if report is None:
            return (
                f"Build halted: {error}. The budget ceiling stopped the run during "
                "analysis, before any task executed. Raise --budget to proceed."
            )
        report.key_decisions.append(f"Halted by budget ceiling: {error}")
        report.finalize()
        report.apply_llm_usage(project.llm_client.cumulative_usage)
        self._persist_learning(project, report)
        report.save(project.path)
        return report.to_markdown()

    def _run_promotion_async(self, project_path) -> None:
        import threading
        from misterdev.core.evolution.tool_promotion import run_tool_promotion

        def _promote():
            try:
                run_tool_promotion(project_path)
            except Exception as e:
                logger.debug(f"Background tool promotion skipped: {e}")

        threading.Thread(target=_promote, daemon=True).start()

    def _run_pipeline(
        self,
        project: Project,
        prompt: str,
        mode: BuildMode,
        flags: BuildFlags,
        assessment: ProjectAssessment,
        env_activate: Optional[str],
        report: BuildReport,
        confirm_plan: bool = False,
        reference_digest: str = "",
        progress_cb: Optional[Callable[..., None]] = None,
        spec_text: str = "",
    ) -> str:
        """Phases 1.5-6: probes, spec, decompose, (confirm), execute, validate.

        Shared by build() and interactive_plan(). When confirm_plan is set, the
        composed plan is shown and the user is asked to approve it before any
        task executes.
        """
        _check_golden_config(project.config)
        # Cross-run env memory: pre-tune this run from durable facts learned in
        # prior runs (effective setup/healthcheck command, a backed-off max_workers)
        # WITHOUT overriding any explicit project.yaml value. Best-effort: a missing
        # or unreadable ledger just means no pre-tuning. Applied here, before any
        # gate/worktree code reads the config.
        try:
            applied = EnvLearnings.load(project.path).apply_to_config(project.config)
            for a in applied:
                logger.info(f"Env-memory: pre-tuned {a} from a prior run")
        except Exception as e:
            logger.debug(f"Env-memory pre-tune skipped (non-fatal): {e}")
        # Make the analysis baseline failure count available to the per-task test
        # gate so a RED baseline doesn't reject every task: the gate then accepts a
        # task that leaves the suite no worse, letting a multi-failure project be
        # fixed incrementally. 0 on a green/unknown baseline keeps the gate strict.
        project.baseline_test_failures = int(
            getattr(assessment.health, "test_failures", 0) or 0
        )
        # Also surface the baseline failure OUTPUT so a fix task can see the real
        # failures on its FIRST attempt (instead of editing blind and only learning
        # what broke on retry). Empty on a green baseline.
        project.baseline_test_output = (
            getattr(assessment.health, "test_output", "") or ""
            if project.baseline_test_failures
            else ""
        )
        # Record the pre-build HEAD so the optional goal-completion check can diff
        # the whole build's work (committed task commits + working tree) against
        # it. Best-effort: None outside a git repo or on error, which the check
        # treats as "no diff" and degrades to a summary-only judgment.
        goal_check_base = self._capture_head(project)
        # Spec-as-tests, when enabled, is wired per-task in the executor: the
        # generated failing test is written under .orchestrator/spec_tests/
        # (outside the project suite, so it never flips the integration-gate
        # baseline) and run scoped after the task's own gates pass. Nothing to do
        # at the pipeline level.
        # Sovereign Phase 1.5: Empirical Probes (only for SMART/CREATE modes).
        # Best-effort: probe discovery must never crash the build, so any
        # failure here degrades to no verified facts rather than aborting. Opt out
        # via orchestrator.enable_probes for a cheaper/faster run (probes spend an
        # LLM call plus ephemeral script runs before any task executes).
        verified_facts = ""
        probes_on = get_setting(project.config, "orchestrator", "enable_probes")
        if probes_on and mode in (BuildMode.SMART, BuildMode.CREATE):
            logger.info("Phase 1.5: Empirical Probe Discovery")
            try:
                probe_gen = ProbeGenerator(project.llm_client)
                with EphemeralCodeManager(project.path) as ephemeral:
                    probes = probe_gen.generate_probes(prompt, assessment.summary())
                    probe_findings = []
                    for p in probes:
                        success, output = ephemeral.run_ephemeral_script(
                            p.get("script", ""),
                            name=f"probe_{p.get('name', 'unknown')}",
                        )
                        probe_findings.append(
                            f"Probe: {p.get('name', '?')} -> {output}"
                        )
                    verified_facts = "\n".join(probe_findings)
            except Exception as e:
                logger.warning(f"Probe discovery failed (non-fatal): {e}")
                report.degraded_subsystems.append(f"Empirical probes: {e}")

        # Verify completeness claims against the real source before they shape the
        # spec or the task list, so deliberate design (graceful-degradation,
        # platform no-ops, parity shims) is not planned as work. Best-effort and
        # conservative: only claims the verifier refutes with evidence are dropped.
        self._verify_completeness_claims(project, assessment, report)

        # Phase 2: Generate Spec (skip when caller supplies one directly)
        spec = (
            spec_text
            if spec_text
            else self._generate_spec(
                mode, prompt, assessment, project, facts=verified_facts
            )
        )

        # Prepend the reference-implementation digest (when porting from one) so
        # decomposition and every task see the reference's real module/symbol map
        # alongside the spec. Read-only and offline; empty when no --reference.
        if reference_digest:
            spec = f"{reference_digest}\n\n{spec}"

        # Sovereign enhancements (metacognition, AB-MCTS) are best-effort: they
        # refine the spec but must not crash the build, so each degrades to the
        # current spec on failure rather than aborting before any work is done.
        # One embedder, shared by lesson retrieval and warm-start, so a similar
        # lesson/task surfaces by MEANING, not just shared tokens. Best-effort:
        # None (no fastembed / disabled) degrades both to lexical ranking.
        embedder = self._learning_embedder(project)
        auditor = SessionAuditor(project.path, project.llm_client, embedder=embedder)
        try:
            # Bias retrieval toward lessons relevant to this build's goal.
            lessons = auditor.get_lessons_context(prompt)
            if lessons:
                spec = f"{lessons}\n\n{spec}"
        except Exception as e:
            logger.warning(f"Lesson injection failed (non-fatal): {e}")
            report.degraded_subsystems.append(f"Lesson injection: {e}")

        # Warm-start: seed the spec with how similar tasks were solved before, so a
        # recurring shape starts from a proven approach instead of cold.
        solved_index = SolvedTaskIndex(
            project.path / ".orchestrator" / "solved_tasks.jsonl", embedder=embedder
        )
        try:
            priors = solved_index.context(prompt)
            if priors:
                spec = f"{priors}\n\n{spec}"
        except Exception as e:
            logger.warning(f"Warm-start injection failed (non-fatal): {e}")
            report.degraded_subsystems.append(f"Warm-start injection: {e}")

        # AB-MCTS branch-and-evaluate is off by default: it fires several serial
        # LLM calls before any work begins (observed ~30 min on one build) for
        # marginal spec refinement. Opt in with orchestrator.enable_ab_mcts.
        if get_setting(project.config, "orchestrator", "enable_ab_mcts"):
            try:
                planner = ABMCTSPlanner(project.llm_client)
                spec = planner.branch_and_evaluate(spec, assessment.summary())
            except Exception as e:
                logger.warning(f"AB-MCTS planning failed (non-fatal): {e}")
                report.degraded_subsystems.append(f"AB-MCTS planning: {e}")

        # Phase 3: Decompose
        max_tasks = get_setting(project.config, "build", "max_tasks")
        # A --max-tasks flag is a per-run ceiling: the tighter of it and the
        # config cap wins, so a focused/bounded run can't balloon into dozens of
        # tasks (and exhaust the budget) the way a broad spec otherwise would.
        if isinstance(flags.max_tasks, int) and flags.max_tasks > 0:
            max_tasks = min(max_tasks, flags.max_tasks)
        # The decomposer otherwise guesses file paths from feature/test names and
        # the executor then CREATES a wrong new file. Feed it the real file+symbol
        # map so a task targets the actual file that defines the code to change.
        file_map = self._project_file_map(project)
        targets = self._resolve_targets(project)
        tasks = decompose_spec(
            spec,
            assessment,
            mode,
            project.llm_client,
            str(project.path),
            max_tasks=max_tasks,
            file_map=file_map,
            targets=targets,
            staging_hint=self._staging_hint(project),
        )
        _early_progress = ProgressTracker(project.path)
        _pre_split_ids = {t.id for t in tasks}
        tasks = _split_oversized_tasks(tasks)
        tasks = split_keystone_tasks(
            tasks, completed_ids=frozenset(_early_progress.completed)
        )
        # Persist any split mappings so needs_rerun() can skip re-executing a
        # segment whose parent was already completed on a prior run.
        _post_ids = {t.id for t in tasks}
        for _orig_id in _pre_split_ids - _post_ids:
            _parts = [
                t.id
                for t in tasks
                if t.id.startswith(f"{_orig_id}-seg")
                or t.id.startswith(f"{_orig_id}-part")
            ]
            if _parts:
                _early_progress.record_split(_orig_id, _parts)
        tasks = topological_sort(tasks)

        if flags.dry_run:
            return format_plan(tasks, mode)

        if confirm_plan:
            _console.print(Markdown(format_plan(tasks, mode)))
            if not self._confirm(f"Proceed with these {len(tasks)} tasks?"):
                return "Cancelled: plan not approved."

        # Phases 4-5: Execute + Gate, wrapped in an outer convergence loop.
        # The loop keeps re-attempting concrete gate failures until the gate is
        # green, the budget runs out, or an iteration makes no progress. The
        # default "auto" is budget-driven: it runs up to CONVERGENCE_CEILING but
        # in practice stops on the budget/no-progress guards below. An explicit
        # positive int caps the iterations hard instead.
        raw_iterations = get_setting(
            project.config, "orchestrator", "max_build_iterations"
        )
        if (
            isinstance(raw_iterations, int)
            and not isinstance(raw_iterations, bool)
            and raw_iterations > 0
        ):
            max_build_iterations = raw_iterations
        else:
            max_build_iterations = CONVERGENCE_CEILING
        iteration = 0
        prev_issues: Optional[list[str]] = None
        while True:
            iteration += 1
            tasks_this_iter = len(tasks)

            # Phase 4: Execution
            self._execute_tasks(tasks, project, flags, report, progress_cb=progress_cb)

            # Phase 5: Gates
            if flags.no_verify:
                # No gate to converge on; preserve single-pass behavior.
                break
            gatekeeper = GateKeeper(
                project.path,
                env_activate=env_activate,
                build_timeout=get_setting(project.config, "build", "build_timeout"),
                test_timeout=get_setting(project.config, "build", "test_timeout"),
                flaky_reruns=get_setting(
                    project.config, "orchestrator", "flaky_reruns"
                ),
                lint_timeout=get_setting(project.config, "build", "lint_timeout"),
                lsp_diagnostics=get_setting(
                    project.config, "orchestrator", "lsp_diagnostics"
                ),
                lsp_language=project.config.get("language"),
                lsp_timeout=get_setting(project.config, "orchestrator", "lsp_timeout"),
                container=self._container_engine(project),
                mutation_gate=get_setting(
                    project.config, "orchestrator", "mutation_gate"
                ),
                mutation_config=project.config.get("mutation") or {},
                runtime_smoke=get_setting(
                    project.config, "orchestrator", "runtime_smoke"
                ),
                runtime_config=project.config.get("runtime") or {},
                web_verify=get_setting(project.config, "orchestrator", "web_verify"),
                vision_verify=get_setting(
                    project.config, "orchestrator", "vision_verify"
                ),
                vision_client=project.llm_client,
            )
            commands = {
                "build_command": assessment.structure.build_command,
                "test_command": assessment.structure.test_command,
                "lint_command": assessment.structure.lint_command
                or detect_lint_command(project.path),
                "audit_command": assessment.structure.audit_command,
                "typecheck_command": project.config.get("typecheck_command")
                or detect_typecheck_command(project.path),
                "golden_command": get_setting(
                    project.config, "orchestrator", "golden_command"
                ),
            }
            success, issues, final_health = gatekeeper.run_gates(commands)
            validation = ValidationResult()
            validation.build_ok = final_health.builds
            validation.tests_ok = final_health.tests_pass
            validation.lint_ok = final_health.lint_clean
            validation.build_ran = bool(commands["build_command"])
            validation.tests_ran = bool(commands["test_command"])
            validation.lint_ran = bool(commands["lint_command"])
            # Per-target validation: verify EVERY declared sub-project with its
            # own toolchain. A target failure fails the run even if the top-level
            # gate was green (no-op when no targets are declared).
            target_results = self._validate_targets(project, env_activate)
            if target_results:
                summary = ", ".join(
                    f"{r['name']}={'OK' if r['ok'] else 'FAIL'}" for r in target_results
                )
                report.key_decisions.append(f"Per-target validation: {summary}")
                failed_targets = [r["name"] for r in target_results if not r["ok"]]
                if failed_targets:
                    issues = list(issues) + [
                        f"Target validation failed: {', '.join(failed_targets)}"
                    ]
                    validation.issues = issues
                    success = False
            report.validation = validation
            report.validation_passed = success
            report.health_after = final_health
            self.last_build_succeeded = success
            if success:
                break
            logger.warning(f"Validation failed: {issues}")
            self._maybe_rollback_regression(project, report, assessment, flags)

            # Decide whether to attempt another convergence iteration.
            if iteration >= max_build_iterations:
                break
            if _budget_exhausted(project.llm_client):
                logger.warning("Convergence halted: LLM budget exhausted.")
                report.key_decisions.append(
                    "Convergence halted: budget exhausted before next iteration"
                )
                break
            # No-progress guards: don't loop on an iteration that ran nothing or
            # produced the identical failure signature.
            if tasks_this_iter == 0:
                break
            if prev_issues is not None and prev_issues == issues:
                report.key_decisions.append(
                    f"Convergence halted: iteration {iteration} reproduced the "
                    "identical gate failures (no progress)"
                )
                break
            prev_issues = list(issues)

            # Build a targeted fix spec from the concrete gate failures plus any
            # failed/deferred tasks, re-decompose it, and re-run the same path.
            fix_spec = self._build_fix_spec(report, issues, final_health)
            fix_tasks = decompose_spec(
                fix_spec,
                assessment,
                mode,
                project.llm_client,
                str(project.path),
                max_tasks=max_tasks,
                file_map=file_map,
                targets=targets,
            )
            tasks = topological_sort(fix_tasks)
            report.key_decisions.append(
                f"Convergence iteration {iteration + 1}: re-attempting "
                f"{len(issues)} gate failure(s) with {len(tasks)} fix task(s)"
            )
            if not tasks:
                report.key_decisions.append(
                    f"Convergence halted: iteration {iteration + 1} produced no "
                    "fix tasks"
                )
                break

        # Optional goal-completion check (off by default). Runs AFTER the gate
        # loop has settled: an LLM judge reads the goal, acceptance criteria, and
        # the cumulative diff and reports whether the goal is actually met (gates
        # green != goal met). Advisory by default — gaps are recorded and logged
        # but do not fail the build; it blocks only when block_on_goal_gap is set.
        # Timeout-bounded and best-effort, so it can never hang or crash a build.
        if get_setting(project.config, "orchestrator", "goal_check"):
            self._run_goal_check(project, prompt, tasks, goal_check_base, report)

        # Phase 6: Metacognitive Audit (best-effort; never fail a finished build)
        report.finalize()
        try:
            auditor.audit_session(
                report.completed_tasks, report.failed_tasks, str(report.scratchpad)
            )
        except Exception as e:
            logger.warning(f"Session audit failed (non-fatal): {e}")
            report.degraded_subsystems.append(f"Session audit: {e}")

        report.apply_llm_usage(project.llm_client.cumulative_usage)
        report.cost_by_task = dict(getattr(project.llm_client, "cost_by_task", {}))
        # Learning writes + spend accounting, shared with the budget-halt path so
        # a budget-exhausted run (the highest-signal failure) is not silently
        # dropped from the streams the self-improvement features learn from.
        self._persist_learning(project, report)

        report.save(project.path)
        return report.to_markdown()
