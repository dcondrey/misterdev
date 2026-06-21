import concurrent.futures
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from my_project_orchestrator.core.registry import ProjectRegistry
from my_project_orchestrator.core.modes import (
    BuildMode,
    BuildFlags,
    parse_flags,
    resolve_mode,
)
from my_project_orchestrator.core.assessment import ProjectAssessment, HealthCheck
from my_project_orchestrator.core.scratchpad import Scratchpad
from my_project_orchestrator.core.decomposer import (
    decompose_spec,
    topological_sort,
    format_plan,
)
from my_project_orchestrator.core.validator import ValidationResult
from my_project_orchestrator.core.gatekeeper import SOTAGateKeeper
from my_project_orchestrator.core.sovereign import (
    StrategyOptimizer,
    RealTimeAligner,
    ABMCTSPlanner,
    EphemeralCodeManager,
    ProbeGenerator,
)
from my_project_orchestrator.core.metacognition import SessionAuditor
from my_project_orchestrator.core.contracts import ContractRegistry
from my_project_orchestrator.core.preflight import PreflightValidator
from my_project_orchestrator.core.progress import ProgressTracker, compute_task_hash
from my_project_orchestrator.core.change_tracker import ChangeTracker
from my_project_orchestrator.core.report import BuildReport
from my_project_orchestrator.core.project import Project
from my_project_orchestrator.core.models import Task
from my_project_orchestrator.analyzers.project_analyzer import analyze_project
from my_project_orchestrator.core.advisor import recommend_work
from my_project_orchestrator.task_executors.markdown_plan_executor import (
    MarkdownPlanExecutor,
)
from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)
console = Console()

MAX_CONSECUTIVE_FAILURES = 3


class _WorktreeProjectView:
    """A Project facade that overrides only `path` (for worktree execution).

    Everything else (config, llm_client, topography, tools, env) delegates to
    the base project, so the executor reads shared context but writes, builds,
    and commits inside the worktree.
    """

    def __init__(self, base, path):
        self._base = base
        self.path = path

    def __getattr__(self, name):
        return getattr(self._base, name)


class ProgressReporter:
    """Lightweight wave/task progress logger for long runs."""

    def __init__(self, total_tasks: int):
        self.total = total_tasks
        self.completed = 0
        self.failed = 0
        self.current_wave = 0
        self.start_time = time.time()
        self._task_start: Optional[float] = None

    def start_wave(self, wave_num: int, task_ids: list[str]):
        self.current_wave = wave_num
        logger.info(f"=== Wave {wave_num} === [{', '.join(task_ids)}]")

    def start_task(self, task_id: str, title: str):
        self._task_start = time.time()
        logger.info(
            f"[{self.completed + self.failed}/{self.total}] Starting {task_id}: {title}"
        )

    def end_task(self, task_id: str, success: bool):
        elapsed = time.time() - self._task_start if self._task_start else 0
        if success:
            self.completed += 1
            logger.info(
                f"[{self.completed + self.failed}/{self.total}] {task_id} DONE ({elapsed:.0f}s)"
            )
        else:
            self.failed += 1
            logger.warning(
                f"[{self.completed + self.failed}/{self.total}] {task_id} FAILED ({elapsed:.0f}s)"
            )

    def summary(self):
        total_time = time.time() - self.start_time
        logger.info(
            f"=== Complete: {self.completed} done, {self.failed} failed, {total_time:.0f}s total ==="
        )


class ProjectOrchestrator:
    """Main orchestrator with Sovereign Grounded SOTA workflow."""

    def __init__(self):
        self.registry = ProjectRegistry()
        self.last_build_succeeded = True

    def scan_directory(self, path: str | Path):
        self.registry.discover_projects(path)

    def list_projects(self) -> Dict[str, Any]:
        return self.registry.list_projects()

    def get_project_status(self, project_path: str | Path) -> Dict[str, Any]:
        project = self.registry.get_project(project_path)
        if not project:
            try:
                project = self.registry.register_project(project_path)
            except Exception as e:
                logger.error(f"Failed to load project at {project_path}: {e}")
                return {"error": f"Project load failed: {e}"}
        project.task_manager.discover_tasks()
        return {
            "name": project.name,
            "path": str(project.path),
            "description": project.description,
            "tasks": [
                {"id": t.id, "status": t.status, "description": t.description[:100]}
                for t in project.task_manager.tasks.values()
            ],
        }

    def run_project(
        self,
        project_path: str | Path,
        dry_run: bool = False,
        skip_preflight: bool = False,
        force: bool = False,
        status: bool = False,
    ):
        """Run pending devplan tasks with dependency-aware orchestration.

        Unlike build(), this executes a pre-written devplan: it skips the
        analysis/spec/decomposition/SOTA-gate phases but adds topological
        ordering, progress-based crash recovery, contract injection, scratchpad
        learning, and change tracking around the existing markdown tasks.
        """
        project = self._get_or_register(project_path)
        if not project:
            return
        if project.env_manager:
            project.env_manager.setup()
        project.task_manager.discover_tasks()
        pending = project.task_manager.get_pending_tasks()
        if not pending:
            logger.info(f"No pending tasks for {project.name}.")
            return
        tasks = topological_sort(pending)

        if not skip_preflight:
            issues = PreflightValidator().validate(tasks, project.path)
            for issue in issues:
                log = logger.error if issue.severity == "error" else logger.warning
                log(f"Preflight {issue.severity}: {issue.task_id}: {issue.message}")
            if PreflightValidator.has_errors(issues):
                logger.error(
                    "Preflight validation failed; aborting. Use --skip-preflight to override."
                )
                return

        if dry_run:
            self._print_execution_plan(tasks)
            return

        if status:
            self._print_rerun_status(
                tasks, ProgressTracker(project.path), project.path, force
            )
            return

        scratchpad = Scratchpad()
        contracts = ContractRegistry(project.path)
        progress = ProgressTracker(project.path)
        changes = ChangeTracker(project.path)
        strategy_optimizer = StrategyOptimizer()
        executor = MarkdownPlanExecutor(scratchpad=scratchpad)
        lang = (project.config.get("language") or "python").lower()
        max_consecutive_failures = project.config.get("orchestrator", {}).get(
            "max_consecutive_failures", MAX_CONSECUTIVE_FAILURES
        )

        completed_ids = set(progress.completed)
        failed_ids: set[str] = set()
        consecutive_failures = 0
        if completed_ids:
            logger.info(f"Resuming run: {len(completed_ids)} tasks already completed")
        logger.info(f"Running {len(tasks)} pending tasks for {project.name}.")

        reporter = ProgressReporter(len(tasks))
        remaining = list(tasks)
        aborted = False
        wave = 0
        while remaining and not aborted:
            ready, still_waiting = [], []
            for task in remaining:
                if not force and not progress.needs_rerun(
                    task.id, compute_task_hash(task, project.path)
                ):
                    completed_ids.add(task.id)
                    continue
                if progress.is_done(task.id):
                    logger.info(
                        f"Task {task.id} previously completed but inputs changed; re-running."
                    )
                if any(d in failed_ids for d in task.dependencies):
                    failed_ids.add(task.id)
                    logger.warning(f"Skipping {task.id}: a dependency failed.")
                    continue
                if any(d not in completed_ids for d in task.dependencies):
                    still_waiting.append(task)
                    continue
                ready.append(task)

            if not ready:
                if still_waiting:
                    logger.error(
                        f"Dependency deadlock; unresolved tasks: {[t.id for t in still_waiting]}"
                    )
                break

            wave += 1
            reporter.start_wave(wave, [t.id for t in ready])
            for task in ready:
                self._inject_task_context(
                    task, contracts, changes, strategy_optimizer, project
                )
                reporter.start_task(task.id, task.title or task.description[:50])
                try:
                    result = executor.execute(task, project)
                    task.execution_history.append(result)
                except Exception as e:
                    logger.error(f"Task {task.id} raised: {e}")
                    result = None

                succeeded = result is not None and result.status == "completed"
                reporter.end_task(task.id, succeeded)
                if succeeded:
                    completed_ids.add(task.id)
                    progress.mark_completed(
                        task.id, compute_task_hash(task, project.path)
                    )
                    consecutive_failures = 0
                    modified = task.files_to_modify + task.files_to_create
                    if modified:
                        contracts.extract_contracts(
                            task.id,
                            modified,
                            project.path,
                            project.llm_client,
                            language=lang,
                        )
                        changes.record_task_changes(task.id, modified)
                else:
                    failed_ids.add(task.id)
                    progress.mark_failed(task.id)
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error("Aborting run: too many consecutive failures.")
                        aborted = True
                        break

            remaining = still_waiting

        reporter.summary()

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
        task.processor_data["sota_strategy"] = strategy_optimizer.select_best_strategy(
            task.description, task.category, "", project.llm_client
        )

    def _print_execution_plan(self, tasks: list[Task]) -> None:
        """Print the dependency-ordered wave plan without executing anything."""
        completed: set[str] = set()
        remaining = list(tasks)
        wave = 0
        console.print(f"\n[bold]Execution Plan (dry-run): {len(tasks)} tasks[/]")
        while remaining:
            ready = [
                t for t in remaining if all(d in completed for d in t.dependencies)
            ]
            if not ready:
                console.print(
                    f"[red]Dependency deadlock among: {[t.id for t in remaining]}[/]"
                )
                break
            wave += 1
            console.print(f"\n[bold cyan]Wave {wave}[/] ({len(ready)} parallel):")
            for t in ready:
                deps = f" -> depends on {t.dependencies}" if t.dependencies else ""
                console.print(
                    f"  [{t.id}] {t.title or t.description[:50]} ({t.complexity}, {t.category}){deps}"
                )
                completed.add(t.id)
            remaining = [t for t in remaining if t.id not in completed]
        console.print(f"\n[dim]Total: {len(tasks)} tasks, {wave} waves.[/]\n")

    def _print_rerun_status(
        self, tasks: list[Task], progress, project_path, force: bool
    ) -> None:
        """Show which tasks would run vs skip, based on content hashes."""
        console.print(f"\n[bold]Task status: {len(tasks)} tasks[/]")
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
                console.print(f"  [yellow]RUN [/] {t.id}  ({reason})")
            else:
                console.print(f"  [green]SKIP[/] {t.id}  (unchanged, completed)")
        console.print(f"\n[dim]{run} would run, {len(tasks) - run} would skip.[/]\n")

    def run_task(self, project_path: str | Path, task_id: str):
        """Runs a specific task for a given project."""
        project = self._get_or_register(project_path)
        if not project:
            return
        if project.env_manager:
            project.env_manager.setup()
        project.task_manager.discover_tasks()
        task = project.task_manager.tasks.get(task_id)
        if not task:
            logger.error(f"Task {task_id} not found in {project.name}.")
            return
        # Inject contracts from this task's already-completed dependencies.
        contracts = ContractRegistry(project.path)
        task.processor_data["interface_contracts"] = contracts.get_contracts_for_task(
            task.dependencies
        )
        executor = MarkdownPlanExecutor()
        try:
            result = executor.execute(task, project)
            task.execution_history.append(result)
        except Exception as e:
            logger.error(f"Task {task_id} raised: {e}")
            project.task_manager.update_task_status(task_id, "failed")

    def build(self, project_path: str | Path, args: str = "") -> str:
        project = self._get_or_register(project_path)
        if not project:
            return "Error: could not load project"

        arg_list = args.split() if args else []
        remaining, flags = parse_flags(arg_list)
        prompt = " ".join(remaining)
        mode = resolve_mode(prompt, project.path)

        logger.info(f"Build started: mode={mode.value}, flags={flags}")
        start_time = datetime.now(timezone.utc)

        # Propagate budget to LLM client
        project.llm_client._budget = flags.budget

        # Preflight: fail fast on a retired/misrouted model id before spending
        # the analysis phase on calls that would 404 mid-run.
        if not flags.dry_run:
            ok, detail = project.llm_client.health_check()
            if not ok:
                logger.error(f"Model preflight failed: {detail}")
                return f"Error: {detail}. Set a valid model in config before building."

        env_activate = None
        if project.env_manager:
            project.env_manager.setup()
            env_activate = project.env_manager.activate_command()

        # Phase 1: Analysis
        assessment = analyze_project(
            project.path,
            project.llm_client,
            build_command=project.config.get("build_command"),
            test_command=project.config.get("test_command"),
            lint_command=project.config.get("lint_command"),
            env_activate=env_activate,
        )

        report = BuildReport(mode, project.name, assessment, start_time)
        report.health_before = assessment.health.model_copy()

        return self._run_pipeline(
            project, prompt, mode, flags, assessment, env_activate, report
        )

    def interactive_plan(self, project_path: str | Path, args: str = "") -> str:
        """Analyze the project, recommend work, and compose a plan with the user.

        The entry point for a plain `project-orchestrator` invocation: instead
        of a predefined devplan, it reads the live project state, proposes
        ranked work items, lets the user choose (or type their own goal), then
        composes and confirms the plan before executing.
        """
        project = self._get_or_register(project_path)
        if not project:
            return "Error: could not load project"

        _, flags = parse_flags(args.split() if args else [])
        start_time = datetime.now(timezone.utc)
        project.llm_client._budget = flags.budget

        ok, detail = project.llm_client.health_check()
        if not ok:
            console.print(f"[red]Model preflight failed:[/] {detail}")
            return f"Error: {detail}. Set a valid model in config before building."

        env_activate = None
        if project.env_manager:
            project.env_manager.setup()
            env_activate = project.env_manager.activate_command()

        console.print(f"[bold]Analyzing[/] {project.name} ...")
        assessment = analyze_project(
            project.path,
            project.llm_client,
            build_command=project.config.get("build_command"),
            test_command=project.config.get("test_command"),
            lint_command=project.config.get("lint_command"),
            env_activate=env_activate,
        )
        console.print(Panel(assessment.summary(), title="Current state", expand=False))

        recs = recommend_work(assessment, project.llm_client)
        goal, mode = self._choose_goal(recs)
        if goal is None:
            return "Cancelled: no work selected."

        report = BuildReport(mode, project.name, assessment, start_time)
        report.health_before = assessment.health.model_copy()
        return self._run_pipeline(
            project,
            goal,
            mode,
            flags,
            assessment,
            env_activate,
            report,
            confirm_plan=True,
        )

    def _choose_goal(self, recs: list) -> tuple[Optional[str], BuildMode]:
        """Present recommendations and return the chosen (goal, mode).

        Returns (None, _) if the user quits. A free-text goal resolves its own
        mode; a picked recommendation carries the advisor's work_type.
        """
        if recs:
            console.print("\n[bold]Recommended work:[/]")
            for i, r in enumerate(recs, 1):
                console.print(
                    f"  [cyan]{i}[/]. {r.title} [dim]({r.work_type}) — {r.rationale}[/]"
                )
        console.print("\nEnter a number to pick, type your own goal, or 'q' to quit.")
        choice = Prompt.ask("Goal").strip()
        if not choice or choice.lower() in ("q", "quit"):
            return None, BuildMode.SMART
        if choice.isdigit() and recs:
            idx = int(choice) - 1
            if 0 <= idx < len(recs):
                r = recs[idx]
                return r.title, self._WORK_TYPE_MODES.get(r.work_type, BuildMode.SMART)
            console.print("[yellow]Out of range; treating input as a goal.[/]")
        return choice, resolve_mode(choice, Path("."))

    _WORK_TYPE_MODES = {
        "debug": BuildMode.DEBUG,
        "complete": BuildMode.COMPLETE,
        "feature": BuildMode.SMART,
        "refactor": BuildMode.SMART,
        "test": BuildMode.SMART,
        "docs": BuildMode.SMART,
    }

    def _confirm(self, question: str) -> bool:
        """Ask a yes/no question; defaults to no."""
        return Prompt.ask(f"{question} [y/N]", default="n").strip().lower() in (
            "y",
            "yes",
        )

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
    ) -> str:
        """Phases 1.5-6: probes, spec, decompose, (confirm), execute, validate.

        Shared by build() and interactive_plan(). When confirm_plan is set, the
        composed plan is shown and the user is asked to approve it before any
        task executes.
        """
        # Sovereign Phase 1.5: Empirical Probes (only for SMART/CREATE modes).
        # Best-effort: probe discovery must never crash the build, so any
        # failure here degrades to no verified facts rather than aborting.
        verified_facts = ""
        if mode in (BuildMode.SMART, BuildMode.CREATE):
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

        # Phase 2: Generate Spec
        spec = self._generate_spec(
            mode, prompt, assessment, project, facts=verified_facts
        )

        # Sovereign enhancements (metacognition, AB-MCTS) are best-effort: they
        # refine the spec but must not crash the build, so each degrades to the
        # current spec on failure rather than aborting before any work is done.
        auditor = SessionAuditor(project.path, project.llm_client)
        try:
            lessons = auditor.get_lessons_context()
            if lessons:
                spec = f"{lessons}\n\n{spec}"
        except Exception as e:
            logger.warning(f"Lesson injection failed (non-fatal): {e}")

        try:
            planner = ABMCTSPlanner(project.llm_client)
            spec = planner.branch_and_evaluate(spec, assessment.summary())
        except Exception as e:
            logger.warning(f"AB-MCTS planning failed (non-fatal): {e}")

        # Phase 3: Decompose
        tasks = decompose_spec(
            spec, assessment, mode, project.llm_client, str(project.path)
        )
        tasks = topological_sort(tasks)

        if flags.dry_run:
            return format_plan(tasks, mode)

        if confirm_plan:
            console.print(Markdown(format_plan(tasks, mode)))
            if not self._confirm(f"Proceed with these {len(tasks)} tasks?"):
                return "Cancelled: plan not approved."

        # Phase 4: Execution
        self._execute_tasks(tasks, project, flags, report)

        # Phase 5: SOTA Gates
        if not flags.no_verify:
            gatekeeper = SOTAGateKeeper(project.path, env_activate=env_activate)
            commands = {
                "build_command": assessment.structure.build_command,
                "test_command": assessment.structure.test_command,
                "lint_command": assessment.structure.lint_command,
            }
            success, issues, final_health = gatekeeper.run_gates(commands)
            validation = ValidationResult()
            validation.build_ok = final_health.builds
            validation.tests_ok = final_health.tests_pass
            validation.lint_ok = final_health.lint_clean
            validation.build_ran = bool(commands["build_command"])
            validation.tests_ran = bool(commands["test_command"])
            validation.lint_ran = bool(commands["lint_command"])
            validation.issues = issues
            report.validation = validation
            report.validation_passed = success
            report.health_after = final_health
            self.last_build_succeeded = success
            if not success:
                logger.warning(f"SOTA Validation Failed: {issues}")
                self._maybe_rollback_regression(project, report, assessment, flags)

        # Phase 6: Metacognitive Audit (best-effort; never fail a finished build)
        report.finalize()
        try:
            auditor.audit_session(
                report.completed_tasks, report.failed_tasks, str(report.scratchpad)
            )
        except Exception as e:
            logger.warning(f"Session audit failed (non-fatal): {e}")

        usage = project.llm_client.cumulative_usage
        report.llm_calls = usage.call_count
        report.llm_tokens = usage.total_tokens
        report.llm_cost = usage.estimated_cost
        report.llm_cache_read_tokens = usage.cache_read_tokens
        report.cost_by_task = dict(getattr(project.llm_client, "cost_by_task", {}))

        report.save(project.path)
        return report.to_markdown()

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
        commits = []
        for t in report.completed_tasks:
            sha = ex.find_task_commit(project, t.id)
            if sha:
                commits.append((t.id, sha))
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

    def _integration_gate(
        self,
        project: Project,
        executor: MarkdownPlanExecutor,
        test_cmd: str,
        wave_tasks: list[Task],
        timeout: int,
    ) -> list[str]:
        """Run the full suite after a wave; revert task commits that regressed it.

        Returns the task_ids whose commits were reverted (empty if the suite
        still passes). Bisects to the single culprit when possible; if that
        can't isolate it or the tree is still red afterward, reverts the
        remaining wave commits (newest first) to restore a green baseline.
        """
        ok, _ = executor._run_command(project, test_cmd, timeout=timeout)
        if ok:
            return []

        commits = []
        for t in wave_tasks:
            sha = executor.find_task_commit(project, t.id)
            if sha:
                commits.append((t.id, sha))
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

    def _execute_tasks(
        self,
        tasks: list[Task],
        project: Project,
        flags: BuildFlags,
        report: BuildReport,
    ) -> None:
        scratchpad = Scratchpad()
        aligner = RealTimeAligner(project.path)
        contracts = ContractRegistry(project.path)
        progress = ProgressTracker(project.path)
        changes = ChangeTracker(project.path)
        strategy_optimizer = StrategyOptimizer()
        executor = MarkdownPlanExecutor(scratchpad=scratchpad)
        report.scratchpad = scratchpad

        orch_cfg = project.config.get("orchestrator", {})
        max_consecutive_failures = orch_cfg.get(
            "max_consecutive_failures", MAX_CONSECUTIVE_FAILURES
        )

        completed_ids = set(progress.completed)
        failed_ids: set[str] = set()
        consecutive_failures = 0
        aborted = False

        # Register decomposed tasks with the TaskManager so status updates,
        # progress tracking, and contract lookups resolve by ID (otherwise
        # build()'s LLM-decomposed task IDs are unknown to the manager).
        for task in tasks:
            project.task_manager.tasks.setdefault(task.id, task)

        if completed_ids:
            logger.info(f"Resuming build: {len(completed_ids)} tasks already completed")

        lang = (project.config.get("language") or "python").lower()
        remaining = list(tasks)

        # Integration gate: after each wave, re-run the full suite and revert
        # any task whose merge regressed it. Per-task validation passes a task
        # in isolation but can't see cross-module breakage that only surfaces
        # when the whole package is imported together; without this, broken
        # tasks accumulate under later merges until the end-of-build gate.
        test_cmd = report.assessment.structure.test_command
        test_timeout = project.config.get("build", {}).get("test_timeout", 180)
        gate_active = (
            not flags.no_rollback
            and orch_cfg.get("integration_gate", True)
            and bool(test_cmd)
            and (Path(project.path) / ".git").exists()
        )
        if gate_active:
            baseline_ok, _ = executor._run_command(
                project, test_cmd, timeout=test_timeout
            )
            if not baseline_ok:
                logger.info(
                    "Integration gate disabled: baseline suite already failing."
                )
                gate_active = False

        while remaining and not aborted:
            # Find all tasks whose dependencies are satisfied
            ready = []
            still_waiting = []
            for task in remaining:
                # Skip only if this exact task (id AND content hash) already
                # completed. Hash-aware so a freshly decomposed plan that reuses
                # generic ids (T-001...) from a prior build is not wrongly
                # skipped against stale progress state.
                if not progress.needs_rerun(
                    task.id, compute_task_hash(task, project.path)
                ):
                    report.completed_tasks.append(task)
                    completed_ids.add(task.id)
                    continue
                if any(d in failed_ids for d in task.dependencies):
                    report.deferred_tasks.append(task)
                    failed_ids.add(task.id)
                    continue
                if any(d not in completed_ids for d in task.dependencies):
                    still_waiting.append(task)
                    continue
                ready.append(task)

            if not ready:
                for task in still_waiting:
                    report.deferred_tasks.append(task)
                break

            # Prepare tasks with context
            for task in ready:
                strategy = strategy_optimizer.select_best_strategy(
                    task.description,
                    task.category,
                    report.assessment.summary(),
                    project.llm_client,
                )
                task.processor_data["sota_strategy"] = strategy
                task.processor_data["consensus_context"] = (
                    aligner.get_consensus_context()
                )
                task.processor_data["interface_contracts"] = (
                    contracts.get_contracts_for_task(task.dependencies)
                )
                task.processor_data["recent_changes"] = (
                    changes.get_recent_changes_for_files(
                        task.files_to_modify + task.files_to_create
                    )
                )

            if flags.interactive:
                filtered_ready = []
                for task in ready:
                    action = self._interactive_prompt(
                        task, task.processor_data.get("sota_strategy", "iterative")
                    )
                    if action == "quit":
                        aborted = True
                        break
                    elif action == "skip":
                        report.deferred_tasks.append(task)
                    else:
                        filtered_ready.append(task)
                ready = filtered_ready
                if aborted:
                    break

            if not ready:
                remaining = still_waiting
                continue

            # Execute: parallel or sequential
            wave_completed: list[Task] = []
            if flags.parallel and len(ready) > 1:
                logger.info(
                    f"Executing {len(ready)} tasks in parallel: {[t.id for t in ready]}"
                )
                results = self._execute_parallel(ready, executor, project)
            else:
                results = []
                for task in ready:
                    try:
                        result = executor.execute(task, project)
                        results.append((task, result, None))
                    except Exception as e:
                        results.append((task, None, e))

            # Process results
            for task, result, error in results:
                if error:
                    logger.error(f"Task {task.id} raised: {error}")
                    failed_ids.add(task.id)
                    progress.mark_failed(task.id)
                    report.failed_tasks.append(task)
                    consecutive_failures += 1
                elif result.status == "completed":
                    task.execution_history.append(result)
                    completed_ids.add(task.id)
                    progress.mark_completed(
                        task.id, compute_task_hash(task, project.path)
                    )
                    report.completed_tasks.append(task)
                    wave_completed.append(task)
                    consecutive_failures = 0
                    modified = task.files_to_modify + task.files_to_create
                    if modified:
                        contracts.extract_contracts(
                            task.id,
                            modified,
                            project.path,
                            project.llm_client,
                            language=lang,
                        )
                        changes.record_task_changes(task.id, modified)
                    if task.complexity == "architectural":
                        aligner.certify_decision(task.title, task.description)
                else:
                    task.execution_history.append(result)
                    failed_ids.add(task.id)
                    progress.mark_failed(task.id)
                    report.failed_tasks.append(task)
                    consecutive_failures += 1

                if consecutive_failures >= max_consecutive_failures:
                    aborted = True
                    break

            # Integration gate: after this wave's merges, revert any task that
            # regressed the full suite before the next wave builds on top of it.
            if gate_active and wave_completed:
                reverted = self._integration_gate(
                    project, executor, test_cmd, wave_completed, test_timeout
                )
                for tid in reverted:
                    completed_ids.discard(tid)
                    failed_ids.add(tid)
                    progress.mark_failed(tid)
                    obj = next((t for t in wave_completed if t.id == tid), None)
                    if obj is not None and obj in report.completed_tasks:
                        report.completed_tasks.remove(obj)
                        report.failed_tasks.append(obj)
                    report.key_decisions.append(
                        f"Integration gate: {tid} reverted (regressed full suite)"
                    )
                    consecutive_failures += 1
                if reverted and consecutive_failures >= max_consecutive_failures:
                    aborted = True

            remaining = still_waiting

        # Defer any unprocessed tasks
        processed_ids = (
            completed_ids | failed_ids | {t.id for t in report.deferred_tasks}
        )
        for task in tasks:
            if task.id not in processed_ids:
                report.deferred_tasks.append(task)

    def _execute_parallel(
        self, ready: list[Task], executor: MarkdownPlanExecutor, project: Project
    ) -> list:
        """Execute a batch of independent tasks concurrently.

        In "worktree" mode each task runs in its own git worktree so parallel
        edits can't collide; otherwise (default "shared") git branching is
        disabled and tasks share the working tree.
        """
        mode = project.config.get("orchestrator", {}).get("parallel_mode", "shared")
        if mode == "worktree" and (project.path / ".git").exists():
            return self._execute_parallel_worktrees(ready, executor, project)

        results = []
        max_workers = project.config.get("orchestrator", {}).get("max_workers", 4)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(ready), max_workers)
        ) as pool:
            future_to_task = {
                pool.submit(executor.execute, task, project, use_git_branch=False): task
                for task in ready
            }
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append((task, result, None))
                except Exception as e:
                    results.append((task, None, e))
        return results

    def _execute_parallel_worktrees(
        self, ready: list[Task], executor: MarkdownPlanExecutor, project: Project
    ) -> list:
        """Run each task in an isolated git worktree, then merge successes back.

        Worktrees are created and merged serially (git's index/worktree metadata
        is not concurrency-safe); only the task bodies run in parallel.
        """
        import uuid
        from my_project_orchestrator.tools.git_tool import GitTool

        git = GitTool({})
        wt_root = project.path / ".orchestrator" / "worktrees"
        wt_root.mkdir(parents=True, exist_ok=True)
        results: list = []
        prepared: list = []

        for task in ready:
            branch = f"task/{task.id}"
            wt_path = wt_root / f"{task.id}-{uuid.uuid4().hex[:6]}"
            ok, out = git.worktree_add(project, str(wt_path), branch, new_branch=True)
            if ok:
                prepared.append((task, wt_path, branch))
            else:
                logger.error(f"Worktree add failed for {task.id}: {out}")
                results.append(
                    (task, None, RuntimeError(f"worktree add failed: {out}"))
                )

        def run_one(item):
            task, wt_path, branch = item
            view = _WorktreeProjectView(project, wt_path)
            try:
                return (
                    task,
                    executor.execute(task, view, use_git_branch=False),
                    None,
                    wt_path,
                    branch,
                )
            except Exception as e:
                return (task, None, e, wt_path, branch)

        max_workers = project.config.get("orchestrator", {}).get("max_workers", 4)
        raw = []
        if prepared:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(len(prepared), max_workers)
            ) as pool:
                futures = [pool.submit(run_one, item) for item in prepared]
                raw = [f.result() for f in concurrent.futures.as_completed(futures)]

        for task, result, error, wt_path, branch in raw:
            if result is not None and getattr(result, "status", None) == "completed":
                merged, mout = git.merge_worktree(project, branch)
                if not merged:
                    logger.error(f"Worktree merge failed for {task.id}: {mout}")
                    error, result = RuntimeError(f"merge failed: {mout}"), None
            git.worktree_remove(project, str(wt_path))
            results.append((task, result, error))
        return results

    def _interactive_prompt(self, task: Task, strategy: str = "iterative") -> str:
        console.print(
            f"\n[bold cyan]Next Task:[/] [{task.id}] {task.title} ([bold magenta]{strategy.upper()}[/])"
        )
        choice = Prompt.ask("Proceed?", choices=["y", "n", "s", "q"], default="y")
        return {"y": "proceed", "q": "quit", "s": "skip", "n": "quit"}[choice]

    def _generate_spec(
        self,
        mode: BuildMode,
        prompt: str,
        assessment: ProjectAssessment,
        project: Project,
        facts: str = "",
    ) -> str:
        """Phase 2: Generate a spec based on mode."""
        if mode == BuildMode.DEBUG:
            parts = ["# Debug Spec\n## Broken Items"]
            for item in assessment.features.broken:
                parts.append(f"- {item}")
            if assessment.features.stubs:
                parts.append("\n## Stubs")
                for item in assessment.features.stubs:
                    parts.append(f"- {item}")
            if not assessment.health.builds:
                parts.append(
                    f"\n## Build Failure\n{assessment.health.build_output[:500]}"
                )
            if not assessment.health.tests_pass:
                parts.append(
                    f"\n## Test Failures\n{assessment.health.test_output[:500]}"
                )
            return "\n".join(parts)

        if mode == BuildMode.COMPLETE:
            parts = [
                f"# Completion Spec\n## Project: {project.name}",
                "## Goal: Complete all work\n",
                "### Must Complete",
            ]
            for f in assessment.features.incomplete:
                parts.append(f"- {f.name}: {f.description}")
            parts.append("\n### Must Fix")
            for item in assessment.features.broken:
                parts.append(f"- {item}")
            for item in assessment.features.stubs:
                parts.append(f"- Stub: {item}")
            parts.append("\n### Should Add")
            for f in assessment.features.missing:
                parts.append(f"- {f.name}: {f.description}")
            if assessment.features.todos:
                parts.append(f"\n### TODOs ({len(assessment.features.todos)} items)")
                for todo in assessment.features.todos[:20]:
                    parts.append(
                        f"- {todo.get('file', '?')}:{todo.get('line', '?')} {todo.get('text', '')}"
                    )
            return "\n".join(parts)

        if mode == BuildMode.SPEC:
            spec_path = project.path / prompt.strip()
            if spec_path.exists():
                return spec_path.read_text(encoding="utf-8")
            return f"Spec file not found: {prompt}"

        if mode in (BuildMode.CREATE, BuildMode.SMART):
            expand_prompt = (
                f"Expand the following into a comprehensive project spec.\n"
                f"Include: features with acceptance criteria, error handling, "
                f"input validation, testing strategy, architecture decisions.\n\n"
                f"Project context: {assessment.context.purpose}\n"
                f"Conventions: {assessment.context.conventions}\n"
                f"Languages: {assessment.structure.languages}\n"
                f"Frameworks: {assessment.structure.frameworks}\n"
                f"Existing features: {[f.name for f in assessment.features.existing]}\n"
                f"Verified facts: {facts}\n\n"
                f"Description: {prompt}\n\nReturn the spec as markdown."
            )
            return project.llm_client.generate_code(
                expand_prompt,
                "You are a software architect writing a project specification.",
            )

        if mode == BuildMode.REVIEW:
            return (
                f"# Review Spec\nReview and fix all issues found in the project.\n"
                f"Broken: {assessment.features.broken}\n"
                f"Stubs: {assessment.features.stubs}\n"
                f"TODOs: {len(assessment.features.todos)} items"
            )

        return f"# Auto Spec\n{prompt}"

    def _get_or_register(self, project_path: str | Path) -> Optional[Project]:
        project = self.registry.get_project(project_path)
        if not project:
            try:
                project = self.registry.register_project(project_path)
            except Exception as e:
                logger.error(f"Failed to register project at {project_path}: {e}")
                return None
        return project
