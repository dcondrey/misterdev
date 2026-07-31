import json
import re
import shlex
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from misterdev.core.execution.registry import ProjectRegistry
from misterdev.core.modes import (
    BuildMode,
    BuildFlags,
    parse_flags,
    resolve_mode,
)
from misterdev.core.planning.assessment import (
    ProjectAssessment,
    HealthCheck,
)
from misterdev.core.context.scratchpad import Scratchpad
from misterdev.core.planning.decomposer import (
    decompose_spec,
    _split_oversized_tasks,
    split_keystone_tasks,
    topological_sort,
    format_plan,
)
from misterdev.core.verification.validator import ValidationResult
from misterdev.core.verification.gatekeeper import GateKeeper
from misterdev.core.gitcmd import run_git
from misterdev.core.planning.sovereign import (
    StrategyOptimizer,
    ABMCTSPlanner,
    EphemeralCodeManager,
    ProbeGenerator,
)
from misterdev.core.planning.metacognition import SessionAuditor
from misterdev.core.learning import SolvedTaskIndex
from misterdev.core.context.contracts import ContractRegistry
from misterdev.core.verification.preflight import PreflightValidator
from misterdev.core.execution.progress import (
    ProgressTracker,
    compute_task_hash,
)
from misterdev.core.context.change_tracker import ChangeTracker
from misterdev.core.reporting.report import BuildReport
from misterdev.core.execution.project import Project
from misterdev.core.execution.deferral import DeferralBook
from misterdev.core.execution.env_learnings import EnvLearnings
from misterdev.core.execution.parallel import ParallelExecutionMixin
from misterdev.core.execution.integration_gate import IntegrationGateMixin
from misterdev.core.execution.doctor_mixin import DoctorMixin
from misterdev.core.execution.targets_mixin import TargetsMixin
from misterdev.core.execution.spec_gen_mixin import SpecGenMixin
from misterdev.core.execution.wave_mixin import WaveMixin
from misterdev.core.execution.interactive_mixin import InteractiveMixin
from misterdev.core.execution.analysis_mixin import AnalysisMixin
from misterdev.core.execution.goal_check_mixin import GoalCheckMixin
from misterdev.core.execution.reporting_mixin import ReportingMixin
from misterdev.core.execution.execution_loop_mixin import ExecutionLoopMixin
from misterdev.core.models import Task
from misterdev.analyzers.project_analyzer.detection import (
    detect_lint_command,
    detect_typecheck_command,
)
from misterdev.analyzers.reference_digest import build_reference_digest
from misterdev.core.planning.advisor import recommend_work
from misterdev.core.planning.plan_store import (
    approved_items,
    save_plan,
)
from misterdev.llm.client import BudgetExceededError
from misterdev.task_executors.markdown_plan_executor import (
    MarkdownPlanExecutor,
)
from misterdev.agent_helpers import (
    ProgressReporter,
    _apply_budget_ceiling,
    _budget_exhausted,
    _check_golden_config,
    _combine_commands,
    _warn_if_baseline_broken,
    _warn_if_no_test_gate,
    _warn_if_test_gate_is_noop,
)
from misterdev.config import get_setting
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)
console = Console()

MAX_CONSECUTIVE_FAILURES = 3

# Safety backstop for "auto" convergence: the loop normally stops earlier on a
# green gate, budget exhaustion, or no-progress, but this bounds a pathological
# run that keeps producing genuinely different (yet still failing) fix tasks.
CONVERGENCE_CEILING = 25


class ProjectOrchestrator(
    DoctorMixin,
    TargetsMixin,
    SpecGenMixin,
    WaveMixin,
    InteractiveMixin,
    AnalysisMixin,
    GoalCheckMixin,
    ReportingMixin,
    ExecutionLoopMixin,
    ParallelExecutionMixin,
    IntegrationGateMixin,
):
    """Main orchestrator with Sovereign Grounded workflow."""

    def __init__(self):
        self.registry = ProjectRegistry()
        self.last_build_succeeded = True
        self.last_build_cost = 0.0
        # Cooperative-stop plumbing for background jobs (see core.execution.jobs).
        # request_stop() lowers the active run's budget so the next model call
        # trips the existing graceful-halt path; None until a run loads a client.
        self._active_client: Any = None
        self._stop_requested = False
        self._stop_lock = threading.Lock()

    def request_stop(self) -> None:
        """Cooperatively cancel the in-flight build/run, if any.

        Reuses the budget kill-switch instead of interrupting the task loop:
        the active client's ceiling is dropped to 0 so its next call raises
        BudgetExceededError, which build()/the pipeline already degrade to a
        partial report. Safe to call before a client exists (the flag is
        honored when the run loads one) and idempotent.
        """
        with self._stop_lock:
            self._stop_requested = True
            client = self._active_client
        if client is not None:
            _apply_budget_ceiling(client, 0.0)

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
        tasklist: Optional[str] = None,
        budget: Optional[float] = None,
        proceed: bool = False,
        progress_cb: Optional[Callable[..., None]] = None,
    ):
        """Run pending devplan tasks with dependency-aware orchestration.

        Unlike build(), this executes a pre-written devplan: it skips the
        analysis/spec/decomposition/gate phases but adds topological
        ordering, progress-based crash recovery, contract injection, scratchpad
        learning, and change tracking around the existing markdown tasks.
        ``tasklist`` points at an external task-list file (any format, possibly
        in another repo) to execute instead of the devplan directory.
        """
        project = self._get_or_register(project_path)
        if not project:
            return
        if tasklist:
            project.config["tasklist"] = tasklist
        # A --budget on `run` is a ceiling like build's: the tighter of it and the
        # project.yaml budget already on the client wins. None -> keep the config
        # budget (this path previously had no CLI override at all).
        if budget is not None:
            _apply_budget_ceiling(project.llm_client, budget)
        # Expose this run's client so a background job's request_stop() can trip
        # the budget kill-switch; honor a stop that arrived before it existed.
        with self._stop_lock:
            self._active_client = project.llm_client
            _stop = self._stop_requested
        if _stop:
            _apply_budget_ceiling(project.llm_client, 0.0)
        _check_golden_config(project.config)
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

        # Requirements preflight: review the WHOLE plan up front for inputs only the
        # user can supply (credentials, accounts, decisions), surface them in
        # REQUIREMENTS.md, and — the smart gate — stop before spending only when a
        # MISSING input would cascade widely. Returns False to stop the run.
        if not self._requirements_preflight(project, tasks, proceed):
            return

        scratchpad = Scratchpad()
        contracts = ContractRegistry(project.path)
        progress = ProgressTracker(project.path)
        changes = ChangeTracker(project.path)
        strategy_optimizer = StrategyOptimizer()
        executor = MarkdownPlanExecutor(scratchpad=scratchpad)
        lang = (project.config.get("language") or "python").lower()
        max_consecutive_failures = get_setting(
            project.config, "orchestrator", "max_consecutive_failures"
        )

        run_parallel = get_setting(project.config, "orchestrator", "run_parallel")
        # The branch a completed/failed task must return to; captured once so a
        # task that strands HEAD (via an unhandled error) can be recovered instead
        # of later tasks piling their merges onto the dead branch.
        _bb = run_git("git rev-parse --abbrev-ref HEAD", project.path)
        base_branch = _bb.stdout.strip() if _bb and _bb.returncode == 0 else None
        completed_ids = set(progress.completed)
        failed_ids: set[str] = set()
        deferred_ids: set[str] = set()
        deferrals: list[dict] = []
        consecutive_failures = 0

        # Walk-away mode: a prior run may have parked tasks with questions in
        # .orchestrator/QUESTIONS.md. Load any answers the user typed and hand each
        # to its task so this run retries it WITH the answer (see execute_mixin).
        deferral_book = DeferralBook(project.path / ".orchestrator")
        answers = deferral_book.load_answers()
        if answers:
            logger.info(f"Loaded {len(answers)} answer(s) from QUESTIONS.md.")
            for t in tasks:
                if t.id in answers:
                    t.processor_data["user_answer"] = answers[t.id]

        if completed_ids:
            logger.info(f"Resuming run: {len(completed_ids)} tasks already completed")
        logger.info(f"Running {len(tasks)} pending tasks for {project.name}.")

        reporter = ProgressReporter(len(tasks))

        def _emit_progress(phase: str) -> None:
            """Best-effort task-level progress to an async job's reporter."""
            if progress_cb is None:
                return
            try:
                progress_cb(done=len(completed_ids), total=len(tasks), phase=phase)
            except Exception as e:  # a progress callback must never break the run
                logger.debug(f"Progress callback failed (non-fatal): {e}")

        _emit_progress("executing")
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
                blocking = [d for d in task.dependencies if d in deferred_ids]
                if blocking:
                    # A dependency is parked awaiting the user, so this task can't
                    # run either — park it too (blocked), never fail, and record a
                    # question that points at the blocker. Keeps the run going.
                    deferred_ids.add(task.id)
                    deferrals.append(
                        {
                            "id": task.id,
                            "title": task.title,
                            "reason": f"blocked by parked task(s): {', '.join(blocking)}",
                            "questions": [
                                f"Answer {', '.join(blocking)} above; this task "
                                "resumes automatically once they do."
                            ],
                        }
                    )
                    continue
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

            def apply_result(task, result) -> bool:
                """Record one task's outcome (parked / done / failed). Returns True
                when the run should abort. Shared by the sequential and parallel
                paths so both handle deferral, progress, contracts, and the
                consecutive-failure guard identically."""
                nonlocal consecutive_failures
                if result is not None:
                    task.execution_history.append(result)
                if result is not None and result.status == "deferred":
                    deferred_ids.add(task.id)
                    deferrals.append(
                        {
                            "id": task.id,
                            "title": task.title,
                            "reason": result.message,
                            "questions": list(result.questions),
                        }
                    )
                    reporter.park_task(task.id, result.message)
                    return False
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
                    _emit_progress(f"wave {wave}")
                    return False
                failed_ids.add(task.id)
                progress.mark_failed(task.id)
                _emit_progress(f"wave {wave}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    logger.error("Aborting run: too many consecutive failures.")
                    return True
                return False

            for task in ready:
                self._inject_task_context(
                    task, contracts, changes, strategy_optimizer, project
                )

            if run_parallel and len(ready) > 1:
                # Independent tasks in a wave run concurrently (worktree-isolated on
                # a git repo). One batch, then outcomes are applied in the SAME
                # deterministic way as the sequential path.
                for task in ready:
                    reporter.start_task(task.id, task.title or task.description[:50])
                try:
                    batch = self._execute_parallel(ready, executor, project)
                except BudgetExceededError as e:
                    logger.warning(f"Halting run: {e}")
                    self._recover_to_base_branch(project, base_branch)
                    aborted = True
                    break
                self._recover_to_base_branch(project, base_branch)
                for task, result, error in batch:
                    if isinstance(error, BudgetExceededError):
                        aborted = True
                        break
                    if error is not None:
                        logger.error(f"Task {task.id} raised: {error}")
                        result = None
                    if apply_result(task, result):
                        aborted = True
                        break
            else:
                for task in ready:
                    reporter.start_task(task.id, task.title or task.description[:50])
                    try:
                        result = executor.execute(task, project)
                    except BudgetExceededError as e:
                        logger.warning(f"Halting run: {e}")
                        aborted = True
                        break
                    except Exception as e:
                        logger.error(f"Task {task.id} raised: {e}")
                        result = None
                    self._recover_to_base_branch(project, base_branch)
                    if apply_result(task, result):
                        aborted = True
                        break

            remaining = still_waiting

        cost = getattr(
            getattr(project.llm_client, "cumulative_usage", None),
            "estimated_cost",
            None,
        )
        reporter.summary(cost=cost)

        # Structured close-out (parity with the build pipeline): classify this run's
        # failures/parks into the taxonomy + write run_summary.json, and record the
        # durable env facts. The `run --tasks` path had neither, so a run on this
        # path was undiagnosable at a glance and never fed P7's cross-run memory.
        try:
            import time

            by_id = {t.id: t for t in tasks}
            failed_items = [
                (tid, self._task_failure_text(by_id[tid]))
                for tid in failed_ids
                if tid in by_id
            ]
            deferred_items = [
                (
                    d["id"],
                    f"{d.get('reason', '')} {' '.join(d.get('questions', []))}".strip(),
                )
                for d in deferrals
            ]
            self._emit_run_summary(
                project,
                len(completed_ids),
                failed_items,
                deferred_items,
                time.time() - reporter.start_time,
            )
            self._record_env_learnings(project)
        except Exception as e:  # a summary/learning write must never sink a run
            logger.warning(f"Run close-out (summary/env-memory) skipped: {e}")

        # Walk-away close-out: if any task parked for the user, write the question
        # book (preserving answers already typed) and surface a concise pointer so
        # a returning user knows exactly what to do to finish the build.
        if deferrals:
            deferral_book.write(deferrals)
            console.print(
                f"\n[yellow]⏸  {len(deferrals)} task(s) need your input.[/] "
                f"Answer them in [bold]{deferral_book.md_path}[/], then re-run the "
                "same command — answered tasks resume, the rest stay parked."
            )
            for d in deferrals[:12]:
                console.print(f"  [dim]{d['id']}[/] {d.get('reason', '')}")
            if len(deferrals) > 12:
                console.print(f"  [dim]... and {len(deferrals) - 12} more[/]")

    def _recover_to_base_branch(self, project, base_branch) -> None:
        """Return HEAD to the run's base branch if a task stranded it.

        The executor creates a ``task/<id>`` branch before running; on a clean
        success or failure it merges/reverts back to base, but an UNHANDLED
        exception can bypass that cleanup and leave HEAD on the dead task branch —
        after which every later sequential task piles its merge onto that branch
        instead of main. Called after each task, this is a no-op when HEAD is
        already on base (the normal case) and a reset-to-base otherwise."""
        if not base_branch:
            return
        proc = run_git("git rev-parse --abbrev-ref HEAD", project.path)
        cur = proc.stdout.strip() if proc and proc.returncode == 0 else ""
        if cur and cur != base_branch:
            run_git("git reset --hard", project.path)
            run_git(f"git checkout {shlex.quote(base_branch)}", project.path)
            logger.warning(
                f"Recovered HEAD from stranded branch '{cur}' back to '{base_branch}'."
            )

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
            console.print(
                f"\n[bold]Requirements review:[/] {len(reqs)} input(s), "
                f"{len(missing)} not yet provided (see "
                f"{project.path / '.orchestrator' / 'REQUIREMENTS.md'})."
            )
            for r in missing[:10]:
                console.print(f"  [yellow]✗[/] {r['key']} — {r.get('summary', '')}")

            gating = gating_requirements(reqs, tasks)
            if gating:
                keys = ", ".join(g["key"] for g in gating)
                console.print(
                    f"[red]Stopping before execution:[/] {keys} are required by "
                    "foundational tasks and would cascade. Provide them (see "
                    "REQUIREMENTS.md), then re-run — or pass [bold]--proceed[/] to "
                    "run now and park what's missing."
                )
                return False
            if missing:
                console.print(
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

    def build(
        self,
        project_path: str | Path,
        args: str = "",
        reference_dir: str | None = None,
        progress_cb: Optional[Callable[..., None]] = None,
        spec_text: str = "",
    ) -> str:
        project = self._get_or_register(project_path)
        if not project:
            return "Error: could not load project"

        arg_list = args.split() if args else []
        remaining, flags = parse_flags(arg_list)
        prompt = " ".join(remaining)
        mode = resolve_mode(prompt, project.path)

        # Optional reference implementation to port from. Validate the path up
        # front (fail fast, like the dirty-tree check) rather than planning
        # against nothing; the digest itself is read-only and offline.
        reference_digest = ""
        if reference_dir:
            try:
                reference_digest = build_reference_digest(
                    reference_dir,
                    cache_dir=project.path / ".orchestrator",
                )
            except (ValueError, OSError) as e:
                logger.error(f"Reference analysis failed: {e}")
                return f"Error: reference dir unusable ({e})."

        logger.info(f"Build started: mode={mode.value}, flags={flags}")
        start_time = datetime.now(timezone.utc)

        # Refuse to run on a dirty working tree: branch-per-task execution
        # carries and can sweep/revert uncommitted changes, so a second writer
        # (or the user's own in-progress work) would be corrupted. Override with
        # --allow-dirty. Skipped for dry-run (read-only).
        if not flags.dry_run and not flags.allow_dirty:
            dirty = self._working_tree_dirty(project)
            if dirty:
                logger.error("Refusing to build on a dirty working tree.")
                return (
                    f"Error: working tree has uncommitted changes ({dirty}). "
                    "Commit or stash them first, or pass --allow-dirty to override."
                )

        # Propagate budget to LLM client
        _apply_budget_ceiling(project.llm_client, flags.budget)

        # Expose this run's client so a background job's request_stop() can trip
        # the budget kill-switch. Honor a stop that arrived before the client
        # existed (dropped budget halts the very first call).
        with self._stop_lock:
            self._active_client = project.llm_client
            _stop = self._stop_requested
        if _stop:
            _apply_budget_ceiling(project.llm_client, 0.0)

        # Preflight: fail fast on a retired/misrouted model id before spending
        # the analysis phase on calls that would 404 mid-run.
        if not flags.dry_run:
            ok, detail = project.llm_client.health_check()
            if not ok:
                logger.error(f"Model preflight failed: {detail}")
                return f"Error: {detail}. Set a valid model in config before building."

        env_activate = self._setup_env(project)

        # The budget ceiling is a graceful kill-switch, not a crash: a
        # BudgetExceededError can surface from ANY model call (an analysis
        # analyzer, probe discovery, spec generation, decomposition, or a task),
        # so wrap the whole pipeline and degrade to a partial report instead of
        # letting the CLI die with a traceback. (Found by dogfooding: a $2 cap was
        # exhausted by pre-execution analysis+probes+spec and crashed the run.)
        report = None
        try:
            # Phase 1: Analysis — skipped when the caller supplies a spec directly
            # (spec_text path: Claude already analysed the codebase; use a zero-cost
            # stub so decomposition still has a well-typed object to read).
            if spec_text:
                assessment = ProjectAssessment()
            else:
                assessment = self._analyze(project, env_activate)

            report = BuildReport(mode, project.name, assessment, start_time)
            report.health_before = assessment.health.model_copy()
            if not spec_text:
                _warn_if_baseline_broken(assessment, report)
                _warn_if_no_test_gate(assessment, project, report)
                _warn_if_test_gate_is_noop(assessment, report)

            result = self._run_pipeline(
                project,
                prompt,
                mode,
                flags,
                assessment,
                env_activate,
                report,
                reference_digest=reference_digest,
                progress_cb=progress_cb,
                spec_text=spec_text,
            )
            self._run_promotion_async(project.path)
            return result
        except BudgetExceededError as e:
            return self._halt_on_budget(project, report, e)

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

    def propose_plan(self, project_path: str | Path, args: str = "") -> Dict[str, Any]:
        """Analyze the project and persist ranked work proposals for approval.

        The non-interactive counterpart to interactive_plan(): runs analysis and
        the advisor, writes the proposals to .orchestrator/proposed_plan.json
        (each unapproved), and returns them so a client can review and approve a
        subset before any code is edited. Spends LLM budget; edits no code.
        """
        project = self._get_or_register(project_path)
        if not project:
            return {"error": "could not load project"}
        _, flags = parse_flags(args.split() if args else [])
        _apply_budget_ceiling(project.llm_client, flags.budget)
        with self._stop_lock:
            self._active_client = project.llm_client
        try:
            env_activate = self._setup_env(project)
            assessment = self._analyze(project, env_activate)
            recs = recommend_work(assessment, project.llm_client)
        except BudgetExceededError as e:
            return {"error": f"budget exhausted during analysis: {e}"}
        return {"items": save_plan(project.path, recs)}

    def execute_plan(self, project_path: str | Path, args: str = "") -> str:
        """Build the approved items from a previously proposed plan.

        Composes a single goal from the approved proposals and runs the normal
        build pipeline (decompose -> execute -> verify) over it. Returns a
        message when nothing is approved. Any build flags in ``args`` (budget,
        parallel, ...) are preserved.
        """
        project = self._get_or_register(project_path)
        if not project:
            return "Error: could not load project"
        approved = approved_items(project.path)
        if not approved:
            return "No approved plan items to execute. Approve items first."
        goal = "complete the following approved work: " + "; ".join(
            it["title"] for it in approved
        )
        build_args = f"{goal} {args}".strip()
        return self.build(project_path, build_args)

    def interactive_plan(self, project_path: str | Path, args: str = "") -> str:
        """Analyze the project, recommend work, and compose a plan with the user.

        The entry point for a plain `misterdev` invocation: instead
        of a predefined devplan, it reads the live project state, proposes
        ranked work items, lets the user choose (or type their own goal), then
        composes and confirms the plan before executing.
        """
        project = self._get_or_register(project_path)
        if not project:
            return "Error: could not load project"

        _, flags = parse_flags(args.split() if args else [])
        start_time = datetime.now(timezone.utc)
        _apply_budget_ceiling(project.llm_client, flags.budget)

        if not flags.allow_dirty:
            dirty = self._working_tree_dirty(project)
            if dirty:
                console.print(
                    f"[red]Working tree has uncommitted changes ({dirty}).[/] "
                    "Commit or stash first, or pass --allow-dirty."
                )
                return f"Error: dirty working tree ({dirty})."

        ok, detail = project.llm_client.health_check()
        if not ok:
            console.print(f"[red]Model preflight failed:[/] {detail}")
            return f"Error: {detail}. Set a valid model in config before building."

        env_activate = self._setup_env(project)

        report = None
        try:
            console.print(f"[bold]Analyzing[/] {project.name} ...")
            assessment = self._analyze(project, env_activate)
            console.print(
                Panel(assessment.summary(), title="Current state", expand=False)
            )

            recs = recommend_work(assessment, project.llm_client)
            goal, mode = self._choose_goal(recs)
            if goal is None:
                return "Cancelled: no work selected."

            report = BuildReport(mode, project.name, assessment, start_time)
            report.health_before = assessment.health.model_copy()
            _warn_if_baseline_broken(assessment, report)
            _warn_if_no_test_gate(assessment, project, report)
            _warn_if_test_gate_is_noop(assessment, report)
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
        except BudgetExceededError as e:
            return self._halt_on_budget(project, report, e)

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

    def _working_tree_dirty(self, project: Project) -> str:
        """Return a short summary if the git working tree has uncommitted changes.

        Returns "" when clean or not a git repo. `git status --porcelain`
        already excludes ignored paths, so the orchestrator's own `.orchestrator/`
        cache (gitignored) never counts as dirty.
        """
        if not (Path(project.path) / ".git").exists():
            return ""
        try:
            proc = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project.path,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (subprocess.SubprocessError, OSError) as e:
            logger.warning(f"Could not check working tree status: {e}")
            return ""
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        if not lines:
            return ""
        return f"{len(lines)} file(s), e.g. {lines[0][3:].strip()}"

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
            console.print(Markdown(format_plan(tasks, mode)))
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

    def _get_or_register(self, project_path: str | Path) -> Optional[Project]:
        project = self.registry.get_project(project_path)
        if not project:
            try:
                project = self.registry.register_project(project_path)
            except Exception as e:
                logger.error(f"Failed to register project at {project_path}: {e}")
                return None
        return project
