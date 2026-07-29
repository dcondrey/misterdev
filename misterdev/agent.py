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
    RealTimeAligner,
    ABMCTSPlanner,
    EphemeralCodeManager,
    ProbeGenerator,
)
from misterdev.core.planning.metacognition import SessionAuditor
from misterdev.core.learning import FailureLog, SolvedTaskIndex
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
from misterdev.utils.file_utils import atomic_write, orchestrator_state_file
from misterdev.core.models import Task
from misterdev.analyzers.project_analyzer import (
    analyze_project,
)
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
    worktree_healthcheck_command,
    worktree_setup_command,
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


class ProjectOrchestrator(ParallelExecutionMixin, IntegrationGateMixin):
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
            from pathlib import Path

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
        # Concise console line — the whole point is one-glance readability.
        mins, secs = divmod(int(summary["elapsed_seconds"]), 60)
        parts = [f"[green]{summary['completed']} done[/]"]
        if summary["deferred"]:
            parts.append(f"[yellow]{summary['deferred']} deferred[/]")
        if summary["failed"]:
            parts.append(f"[red]{summary['failed']} failed[/]")
        console.print(
            "[bold]Run summary[/] · "
            + " · ".join(parts)
            + f" · [dim]{mins}m {secs}s[/]"
        )
        if summary["failure_breakdown"]:
            brk = ", ".join(
                f"{cat} {n}" for cat, n in summary["failure_breakdown"].items()
            )
            console.print(f"  [dim]failures:[/] {brk}")
            top = summary["top_obstacle"]
            if top:
                ex = summary["exemplars"].get(top, "")
                console.print(
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

    def run_doctor(self, project_path: str | Path) -> Dict[str, Any]:
        """Preflight a project for an unattended run: gather the environment facts
        and route them through the pure ``doctor`` checks. Returns the aggregate
        (counts + exit_code) with the per-check results under ``checks``. Never
        raises — a gathering error degrades that one check, it does not crash."""
        from misterdev.core.execution import doctor as dr

        project = self._get_or_register(project_path)
        if not project:
            err = dr.Check(
                "load project", dr.FAIL, "could not load the project", "check the path"
            )
            out = dr.aggregate([err])
            out["checks"] = [err]
            return out

        checks: list = []
        is_git = (Path(project.path) / ".git").exists()
        checks.append(dr.check_git_repo(is_git))
        base = self._doctor_base_branch(project)
        if is_git:
            checks.append(dr.check_clean_tree(self._working_tree_dirty(project)))
            checks.append(
                dr.check_on_base_branch(self._doctor_current_branch(project), base)
            )
            checks.append(
                dr.check_leftover_task_branches(self._doctor_task_branches(project))
            )
            checks.append(dr.check_dangling_worktrees(self._doctor_worktrees(project)))
        try:
            ok, detail = project.llm_client.health_check()
        except Exception as e:  # a gathering error is not a model verdict
            ok, detail = False, str(e)
        checks.append(dr.check_models(ok, detail))
        if is_git:
            checks.append(self._doctor_worktree_probe(project))
        checks.append(
            dr.check_requirements(self._doctor_unsatisfied_requirements(project))
        )

        out = dr.aggregate(checks)
        out["checks"] = checks
        return out

    @staticmethod
    def _doctor_base_branch(project) -> str:
        """The repo's base branch: ``main`` or ``master`` if present, else HEAD."""
        for name in ("main", "master"):
            proc = run_git(
                f"git rev-parse --verify --quiet {shlex.quote(name)}", project.path
            )
            if proc is not None and proc.returncode == 0 and proc.stdout.strip():
                return name
        return ProjectOrchestrator._doctor_current_branch(project) or "main"

    @staticmethod
    def _doctor_current_branch(project) -> Optional[str]:
        proc = run_git("git rev-parse --abbrev-ref HEAD", project.path)
        if proc is None or proc.returncode != 0:
            return None
        return proc.stdout.strip() or None

    @staticmethod
    def _doctor_task_branches(project) -> list:
        proc = run_git("git branch --list task/*", project.path)
        if proc is None or proc.returncode != 0:
            return []
        return [ln.strip(" *").strip() for ln in proc.stdout.splitlines() if ln.strip()]

    @staticmethod
    def _doctor_worktrees(project) -> list:
        """Registered worktrees under the orchestrator's worktree dir (excludes the
        main checkout), i.e. leftovers a prior run did not clean up."""
        proc = run_git("git worktree list --porcelain", project.path)
        if proc is None or proc.returncode != 0:
            return []
        found = []
        marker = str(Path(project.path) / ".orchestrator" / "worktrees")
        for ln in proc.stdout.splitlines():
            if ln.startswith("worktree ") and marker in ln:
                found.append(ln[len("worktree ") :].strip())
        return found

    def _doctor_worktree_probe(self, project):
        """Create a throwaway worktree, prime deps + run the healthcheck, remove it.
        Best-effort — any failure becomes a WARN, never crashes the doctor."""
        from misterdev.core.execution import doctor as dr

        setup_cmd = self._worktree_setup_command(project)
        health_cmd = self._worktree_healthcheck_command(project)
        if not setup_cmd and not health_cmd:
            return dr.check_worktree_prime(None, None)
        import uuid
        from misterdev.tools.command import CommandTool
        from misterdev.tools.git_tool import GitTool

        git = GitTool({})
        wt_root = Path(project.path) / ".orchestrator" / "worktrees"
        wt_root.mkdir(parents=True, exist_ok=True)
        git.worktree_prune(project)
        run_id = uuid.uuid4().hex[:6]
        branch = f"doctor/{run_id}"
        wt_path = wt_root / f"doctor-{run_id}"
        ok, out = git.worktree_add(project, str(wt_path), branch, new_branch=True)
        if not ok:
            return dr.Check(
                "worktree prime + healthcheck",
                dr.WARN,
                f"could not create a throwaway worktree: {out[-120:]}",
                "check git worktree support and disk space",
            )
        timeout = get_setting(project.config, "orchestrator", "worktree_setup_timeout")
        cmd = CommandTool({})
        prime_ok = None
        health_ok = None
        detail = ""
        try:
            if setup_cmd:
                prime_ok, pout = cmd.execute(
                    project, setup_cmd, cwd=str(wt_path), timeout=timeout
                )
                if not prime_ok:
                    detail = pout[-160:]
            if health_cmd:
                health_ok, hout = cmd.execute(
                    project, health_cmd, cwd=str(wt_path), timeout=timeout
                )
                if not health_ok:
                    detail = hout[-160:]
        finally:
            git.worktree_remove(project, str(wt_path))
            git.branch_delete(project, branch)
        return dr.check_worktree_prime(prime_ok, health_ok, detail)

    def _doctor_unsatisfied_requirements(self, project) -> list:
        """Keys in .orchestrator/REQUIREMENTS.md marked MISSING with no typed answer.
        Empty when the file is absent (nothing reviewed yet) or all are provided."""
        md = Path(project.path) / ".orchestrator" / "REQUIREMENTS.md"
        if not md.exists():
            return []
        try:
            from misterdev.core.planning.requirements import RequirementsBook

            answers = RequirementsBook(
                Path(project.path) / ".orchestrator"
            ).load_answers()
            text = md.read_text(encoding="utf-8")
        except OSError:
            return []
        unsatisfied = []
        key = ""
        missing = False
        for line in text.splitlines():
            if line.startswith("## "):
                if key and missing and key not in answers:
                    unsatisfied.append(key)
                key = line[3:].split("—", 1)[0].strip().strip("`")
                missing = False
            elif line.strip().lower().startswith("- status:"):
                missing = "missing" in line.lower()
        if key and missing and key not in answers:
            unsatisfied.append(key)
        return unsatisfied

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

    def _execute_tasks(
        self,
        tasks: list[Task],
        project: Project,
        flags: BuildFlags,
        report: BuildReport,
        progress_cb: Optional[Callable[..., None]] = None,
    ) -> None:
        scratchpad = Scratchpad()
        aligner = RealTimeAligner(project.path)
        contracts = ContractRegistry(project.path)
        progress = ProgressTracker(project.path)
        changes = ChangeTracker(project.path)
        strategy_optimizer = StrategyOptimizer()
        executor = MarkdownPlanExecutor(scratchpad=scratchpad)
        report.scratchpad = scratchpad

        max_consecutive_failures = get_setting(
            project.config, "orchestrator", "max_consecutive_failures"
        )

        completed_ids = set(progress.completed)
        failed_ids: set[str] = set()

        def _emit_progress(phase: str) -> None:
            """Best-effort task-level progress to an async job's reporter."""
            if progress_cb is None:
                return
            try:
                progress_cb(done=len(completed_ids), total=len(tasks), phase=phase)
            except Exception as e:  # a progress callback must never break the build
                logger.debug(f"Progress callback failed (non-fatal): {e}")

        _emit_progress("executing")
        # Skip a ready task before spawning a worktree when it is already
        # satisfied (content hash unchanged since a recorded completion). Default
        # on; false forces every ready task to re-run.
        skip_satisfied = get_setting(
            project.config, "orchestrator", "skip_satisfied_tasks"
        )
        # Adaptive concurrency/timeout backoff: capture the CONFIGURED values as the
        # recovery ceiling, start each run at full concurrency (factor 1.0), and
        # re-tune between waves from each wave's infra-failure count.
        from misterdev.core.execution.adaptive import WaveTuning, next_wave_tuning

        adaptive = get_setting(project.config, "orchestrator", "adaptive_concurrency")
        adaptive_base = {
            "workers": get_setting(project.config, "orchestrator", "max_workers"),
            "setup": get_setting(
                project.config, "orchestrator", "worktree_setup_timeout"
            ),
            "build": get_setting(project.config, "build", "build_timeout"),
            "test": get_setting(project.config, "build", "test_timeout"),
        }
        wave_tuning = WaveTuning(int(adaptive_base["workers"]), 1.0)
        consecutive_failures = 0
        aborted = False
        max_cost_per_task = get_setting(
            project.config, "orchestrator", "max_cost_per_task"
        )

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
        # Fold the golden suite into the per-wave gate so a task that breaks the
        # immutable contract is bisected and reverted immediately, not just at
        # the end-of-iteration GateKeeper.
        test_cmd = _combine_commands(
            report.assessment.structure.test_command,
            get_setting(project.config, "orchestrator", "golden_command"),
        )
        test_timeout = get_setting(project.config, "build", "test_timeout")
        gate_targets = project.config.get("targets") or []
        gate_active = (
            not flags.no_rollback
            and get_setting(project.config, "orchestrator", "integration_gate")
            and (bool(test_cmd) or bool(gate_targets))
            and (Path(project.path) / ".git").exists()
        )
        # Per-target baselines (multi-target): measure each sub-project's gate in
        # its own dir ONCE up front, so the per-wave gate reverts only a target
        # that REGRESSED, and the executor's per-task gate uses the right baseline.
        target_baselines: dict = {}
        if gate_active and gate_targets:
            for tgt in gate_targets:
                gcmd = tgt.get("test_command") or tgt.get("build_command")
                if not gcmd:
                    continue
                tname = tgt.get("name") or tgt.get("path")
                tp = (tgt.get("path") or "").strip("/")
                rdir = project.path / tp if tp else project.path
                target_baselines[tname] = self._suite_failures(
                    project, executor, gcmd, test_timeout, cwd=rdir
                )
            project.target_baselines = {
                k: int(v or 0) for k, v in target_baselines.items()
            }
        baseline_failures = 0
        project.baseline_test_failing_ids = None
        if gate_active and test_cmd:
            baseline_ok, baseline_out = executor._run_command(
                project, test_cmd, timeout=test_timeout
            )
            if not baseline_ok:
                # A red baseline used to fully disable the gate, so a task could
                # WORSEN the suite (more failures than at the start) and still
                # commit. Instead, run the gate in COUNT mode against the baseline
                # failure count when it is parseable — reverting only a wave that
                # increases failures. Unparseable count -> disable as before.
                from misterdev.core.verification.validator import (
                    _parse_test_counts,
                )

                total, fails = _parse_test_counts(baseline_out)
                if total > 0 and fails > 0:
                    baseline_failures = fails
                    # Capture the failing-test IDENTITIES from the same baseline
                    # output so the post-wave gate can prefer identity mode (revert
                    # on any NEW failure) over count mode (revert only on a count
                    # rise); None when unparseable -> count-mode fallback.
                    project.baseline_test_failing_ids = self._failing_ids_from_output(
                        baseline_out, project
                    )
                    mode = (
                        "IDENTITY mode: revert a wave that adds any new failing test"
                        if project.baseline_test_failing_ids
                        else "COUNT mode: revert a wave that raises the count"
                    )
                    logger.info(
                        f"Integration gate in {mode}; baseline has {fails} "
                        "failing test(s)."
                    )
                else:
                    logger.info(
                        "Integration gate disabled: baseline failing and test "
                        "count unparseable."
                    )
                    gate_active = False

        _wave_num = 0
        while remaining and not aborted:
            # Graceful budget stop: if the global budget is exhausted, do not
            # launch another wave. Defer the remainder and break so the pipeline
            # finalizes/reports normally instead of throwing mid-wave.
            if _budget_exhausted(project.llm_client):
                logger.warning("Stopping run: LLM budget exhausted.")
                report.key_decisions.append(
                    "Stopped: budget exhausted; remaining work deferred"
                )
                for task in remaining:
                    report.deferred_tasks.append(task)
                break

            _wave_num += 1
            # Find all tasks whose dependencies are satisfied
            ready = []
            still_waiting = []
            for task in remaining:
                # Skip only if this exact task (id AND content hash) already
                # completed — mark it done WITHOUT spawning a worktree, so an
                # already-satisfied task doesn't pay a prime/install just to be
                # re-recognized. Hash-aware so a freshly decomposed plan that
                # reuses generic ids (T-001...) from a prior build is not wrongly
                # skipped against stale progress state. Gated by
                # skip_satisfied_tasks (default on); no gate is run on the base
                # branch for this — it rests on the content hash plus the ledger.
                if skip_satisfied and not progress.needs_rerun(
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
                task.processor_data["strategy"] = strategy
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
                        task, task.processor_data.get("strategy", "iterative")
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

            # Apply this wave's adapted concurrency/timeouts (a no-op at full
            # tuning) before dispatch, so the worktree/gate code reads the backed-
            # off values under contention.
            if adaptive:
                self._apply_wave_tuning(project, wave_tuning, adaptive_base)

            _wave_cost_before = getattr(
                getattr(project.llm_client, "cumulative_usage", None),
                "estimated_cost",
                0.0,
            )

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
            for i, (task, result, error) in enumerate(results):
                if isinstance(error, BudgetExceededError):
                    # Budget ran out mid-task: revert any partial work and stop
                    # the run gracefully rather than recording a failure.
                    logger.warning(
                        f"Stopping run: budget exhausted during {task.id} ({error})."
                    )
                    sha = executor.find_task_commit(project, task.id)
                    if sha:
                        executor.revert_task_commit(project, sha)
                    report.deferred_tasks.append(task)
                    for rem_task, _, _ in results[i + 1 :]:
                        report.deferred_tasks.append(rem_task)
                    report.key_decisions.append(
                        "Stopped: budget exhausted; remaining work deferred"
                    )
                    aborted = True
                    break
                exceeded_fn = getattr(project.llm_client, "task_cost_exceeded", None)
                if (
                    max_cost_per_task is not None
                    and exceeded_fn
                    and exceeded_fn(task.id)
                ):
                    # This task alone blew its per-task cost cap: abandon and
                    # revert it cleanly instead of burning more budget retrying.
                    cap_fn = getattr(project.llm_client, "effective_task_cap", None)
                    cap = (cap_fn(task.id) if cap_fn else None) or 0.0
                    logger.warning(
                        f"Task {task.id} hit per-task cost cap "
                        f"(${cap:.2f}); abandoning and reverting."
                    )
                    sha = executor.find_task_commit(project, task.id)
                    if sha:
                        executor.revert_task_commit(project, sha)
                    failed_ids.add(task.id)
                    progress.mark_failed(task.id)
                    report.deferred_tasks.append(task)
                    report.key_decisions.append(
                        f"Task {task.id} deferred: exceeded per-task cost cap "
                        f"(${cap:.2f})"
                    )
                    continue
                if error is not None:
                    logger.error(f"Task {task.id} raised: {error}")
                    # Preserve the failure text (merge conflict, worktree add, a
                    # raised exception) so the end-of-run taxonomy can classify it;
                    # this path records no ExecutionResult to read it back from.
                    if isinstance(getattr(task, "processor_data", None), dict):
                        task.processor_data["failure_text"] = str(error)
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
                elif result.status == "deferred":
                    # Parked (walk-away input needed, or escalated to decomposition)
                    # — NOT a code failure, so it must not trip the consecutive-
                    # failure abort or be recorded as terminally failed (a re-run
                    # retries it). It still blocks dependents and feeds the
                    # convergence loop's re-decomposition, like a non-success.
                    task.execution_history.append(result)
                    failed_ids.add(task.id)
                    report.deferred_tasks.append(task)
                    consecutive_failures = 0
                else:
                    task.execution_history.append(result)
                    failed_ids.add(task.id)
                    progress.mark_failed(task.id)
                    report.failed_tasks.append(task)
                    consecutive_failures += 1

                _emit_progress("building")
                if consecutive_failures >= max_consecutive_failures:
                    aborted = True
                    break

            _wave_cost = (
                getattr(
                    getattr(project.llm_client, "cumulative_usage", None),
                    "estimated_cost",
                    0.0,
                )
                - _wave_cost_before
            )
            if isinstance(_wave_cost, (int, float)) and _wave_cost > 0.0:
                report.key_decisions.append(f"Wave {_wave_num} cost: ${_wave_cost:.4f}")

            _wave_failed = {
                t.id for t in ready if t.id in failed_ids and t.id not in completed_ids
            }
            for _cf in (t for t in ready if t.id in _wave_failed):
                _ft = self._task_failure_text(_cf).lower()
                if "conflict" not in _ft and "<<<" not in _ft:
                    continue
                _cf_files = set(_cf.files_to_create + _cf.files_to_modify)
                for _partner in ready:
                    if _partner.id == _cf.id:
                        continue
                    if not _cf_files & set(
                        _partner.files_to_create + _partner.files_to_modify
                    ):
                        continue
                    progress.record_conflict(_cf.id, _partner.id)
                    if (
                        progress.conflict_count(_cf.id, _partner.id) >= 2
                        and _cf.id not in _partner.dependencies
                    ):
                        _partner.dependencies.append(_cf.id)
                        logger.info(
                            "Injecting dep %s→%s after repeated conflict on shared files.",
                            _cf.id,
                            _partner.id,
                        )

            # Integration gate: after this wave's merges, revert any task that
            # regressed the full suite before the next wave builds on top of it.
            if gate_active and wave_completed:
                if gate_targets:
                    # Polyglot: gate each touched sub-project with its own
                    # toolchain in its own directory (replaces the top-level gate,
                    # which would use the wrong commands for a frontend wave).
                    reverted = self._integration_gate_targets(
                        project,
                        executor,
                        gate_targets,
                        wave_completed,
                        test_timeout,
                        target_baselines,
                    )
                else:
                    reverted = self._integration_gate(
                        project,
                        executor,
                        test_cmd,
                        wave_completed,
                        test_timeout,
                        baseline_failures=baseline_failures,
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

            # Re-tune concurrency/timeouts for the NEXT wave from THIS wave's infra
            # faults: back off under contention, recover gradually when clean.
            if adaptive:
                infra_count = self._wave_infra_count(results)
                nxt = next_wave_tuning(
                    infra_count,
                    wave_tuning,
                    base_workers=int(adaptive_base["workers"]),
                    threshold=get_setting(
                        project.config, "orchestrator", "adaptive_infra_threshold"
                    ),
                    timeout_factor=get_setting(
                        project.config, "orchestrator", "adaptive_timeout_factor"
                    ),
                    max_timeout_factor=get_setting(
                        project.config, "orchestrator", "adaptive_max_timeout_factor"
                    ),
                )
                if nxt != wave_tuning:
                    logger.info(
                        f"Adaptive tuning: {infra_count} infra fault(s) this wave; "
                        f"next wave workers {wave_tuning.max_workers}->"
                        f"{nxt.max_workers}, timeout x{wave_tuning.timeout_factor:g}"
                        f"->x{nxt.timeout_factor:g}."
                    )
                wave_tuning = nxt

            remaining = still_waiting

        # Restore the configured concurrency/timeouts so nothing downstream sees a
        # backed-off value left over from an infra-heavy wave. Expose the value the
        # run settled on (and the base it started from) so the cross-run env memory
        # can persist a contention-driven reduction for the next run.
        if adaptive:
            project.env_settled_workers = wave_tuning.max_workers
            project.env_base_workers = int(adaptive_base["workers"])
            self._apply_wave_tuning(
                project, WaveTuning(int(adaptive_base["workers"]), 1.0), adaptive_base
            )

        # Defer any unprocessed tasks
        processed_ids = (
            completed_ids | failed_ids | {t.id for t in report.deferred_tasks}
        )
        for task in tasks:
            if task.id not in processed_ids:
                report.deferred_tasks.append(task)

    def _interactive_prompt(self, task: Task, strategy: str = "iterative") -> str:
        console.print(
            f"\n[bold cyan]Next Task:[/] [{task.id}] {task.title} ([bold magenta]{strategy.upper()}[/])"
        )
        choice = Prompt.ask("Proceed?", choices=["y", "n", "s", "q"], default="y")
        return {"y": "proceed", "q": "quit", "s": "skip", "n": "quit"}[choice]

    def _staging_hint(self, project: Project) -> str:
        """Dense-reward staging suggestion for a single complex source file.

        Uses the already-built symbol graph: when every public symbol lives in ONE
        non-test source file, that is a single-file goal — synthesize ordered
        construction->mutation->query stages so the decomposer can split it into a
        few sequential, independently-verifiable sub-tasks (raises per-attempt
        success on state-heavy files). Empty for multi-file goals or when nothing
        stages; never raises.
        """
        try:
            from misterdev.core.planning.verifier_decomposition import (
                render_stages,
                synthesize_stages,
            )

            graph = getattr(getattr(project, "topography", None), "graph", None)
            symbols = list(getattr(graph, "symbols", {}).values()) if graph else []
            src = [
                s
                for s in symbols
                if getattr(s, "file_path", "") and "test" not in s.file_path.lower()
            ]
            if len({s.file_path for s in src}) != 1:
                return ""  # staging only applies to a single-file goal
            stages = synthesize_stages(src)
            if len(stages) < 2:
                return ""
            return (
                "\n## Suggested staging (dense-reward decomposition)\n"
                "This file's public API splits into ordered, independently-"
                "verifiable stages. Prefer ONE sequential sub-task per stage, in "
                "this order (each must compile and leave the suite no worse):\n"
                f"{render_stages(stages)}\n"
            )
        except Exception as e:  # a staging hint must never break decomposition
            logger.debug(f"Staging hint skipped (non-fatal): {e}")
            return ""

    def _ground_completion_spec(
        self, assessment: ProjectAssessment, project: Project
    ) -> str:
        """Build a COMPLETE-mode spec grounded in objective signals.

        A vague "complete everything" goal on a real codebase otherwise churns:
        the completeness analyzer flags "incomplete"/"stub" items from a lossy
        overview and mislabels deliberate design (graceful degradation, platform
        no-ops) as work, so the spec becomes a pile of speculative tasks. Instead
        lead with HARD signals — a failing build, failing tests, located
        TODO/FIXME markers, broken references — which are objective and
        verifiable, add features the docs promise but are absent, and demote the
        analyzer's guesses to an explicit "do NOT task these unless corroborated"
        advisory. When nothing hard or documented exists, the goal is ill-posed:
        emit zero-task guidance rather than fabricate work.
        """
        h = assessment.health
        f = assessment.features
        hard: list[str] = []
        if not h.builds and h.build_output:
            hard.append(
                f"- The build is FAILING; fix it first:\n{h.build_output[:400]}"
            )
        if not h.tests_pass and h.test_output:
            hard.append(f"- Tests are FAILING:\n{h.test_output[:400]}")
        hard.extend(f"- Broken: {item}" for item in f.broken)
        hard.extend(
            f"- {t.get('file', '?')}:{t.get('line', '?')} {t.get('text', '')}"
            for t in f.todos[:20]
        )
        documented = [f"- {m.name}: {m.description}" for m in f.missing]
        speculative = [f"- {i.name}: {i.description}" for i in f.incomplete]
        speculative += [f"- Stub: {s}" for s in f.stubs]

        parts = [f"# Completion Spec\n## Project: {project.name}\n"]
        if hard:
            parts.append(
                "## Must Fix — objective, verifiable failures\n" + "\n".join(hard)
            )
        if documented:
            parts.append(
                "\n## Should Add — promised by the docs but absent\n"
                + "\n".join(documented)
            )
        if not hard and not documented:
            parts.append(
                "## No concrete objective found\n"
                "The build and tests pass and there are no TODO/FIXME markers or "
                "documented-but-missing features. A vague 'complete everything' goal "
                "has no well-posed work here. Do NOT fabricate tasks from speculation: "
                "produce ZERO tasks and report that a specific objective (a feature, a "
                "bug to fix, or --focus <area>) is required."
            )
        if speculative:
            parts.append(
                "\n## Advisory — analyzer guesses, NOT tasks\n"
                "Inferred as incomplete/stub from a lossy overview; these often "
                "mislabel deliberate design. Do NOT create a task for any of these "
                "unless a failing test or build error above corroborates it.\n"
                + "\n".join(speculative[:15])
            )
        return "\n".join(parts)

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
            return self._ground_completion_spec(assessment, project)

        if mode == BuildMode.SPEC:
            spec_path = project.path / prompt.strip()
            try:
                spec_path = spec_path.resolve()
                spec_path.relative_to(project.path.resolve())
            except ValueError:
                return f"Spec file not found: {prompt}"
            if spec_path.exists():
                return spec_path.read_text(encoding="utf-8")
            return f"Spec file not found: {prompt}"

        if mode == BuildMode.CREATE:
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

        if mode == BuildMode.SMART:
            # SMART is a SPECIFIC instruction on an EXISTING project — not a
            # from-scratch build. The goal is the scope boundary: implement
            # exactly what it asks plus only what is strictly necessary to make
            # THAT correct and tested. Never expand into a whole-project spec
            # (that is CREATE's job) — doing so makes the decomposer invent
            # unrelated tasks and rewrite pre-existing files it was only meant
            # to read (observed: a "create region.py" goal ballooned into
            # rewriting the harness and inventing conftest/config tasks).
            scoped_prompt = (
                f"Write a tightly-scoped implementation spec for EXACTLY this "
                f"goal — nothing more.\n"
                f"Rules:\n"
                f"- Implement only what the goal asks. Add only what is strictly "
                f"necessary to make the goal correct, tested, and safe (its own "
                f"error handling, input validation, and tests).\n"
                f"- Do NOT expand scope: no unrelated features, no 'completing' "
                f"or 'improving' the project, no refactors the goal did not "
                f"request.\n"
                f"- Existing files are CONTEXT, not work items. Do not modify or "
                f"rewrite them unless the goal explicitly requires it.\n"
                f"- Prefer the smallest change set that fully satisfies the "
                f"goal.\n\n"
                f"Project context: {assessment.context.purpose}\n"
                f"Conventions: {assessment.context.conventions}\n"
                f"Languages: {assessment.structure.languages}\n"
                f"Existing files (context only — do not modify unless the goal "
                f"requires it): {[f.name for f in assessment.features.existing]}\n"
                f"Verified facts: {facts}\n\n"
                f"Goal: {prompt}\n\nReturn the spec as markdown."
            )
            return project.llm_client.generate_code(
                scoped_prompt,
                "You are a software engineer writing a tightly-scoped "
                "implementation spec. You implement exactly what is asked and "
                "resist scope creep.",
            )

        if mode == BuildMode.REVIEW:
            return (
                f"# Review Spec\nReview and fix all issues found in the project.\n"
                f"Broken: {assessment.features.broken}\n"
                f"Stubs: {assessment.features.stubs}\n"
                f"TODOs: {len(assessment.features.todos)} items"
            )

        return f"# Auto Spec\n{prompt}"

    def _setup_env(self, project: Project) -> Optional[str]:
        """Initialize the project's env manager and return its activation prefix."""
        if project.env_manager:
            project.env_manager.setup()
            return project.env_manager.activate_command()
        return None

    def _container_engine(self, project: Project):
        """Return the project's container engine if a container environment is
        configured and an engine is available, else None (gates run locally).

        ``_setup_env`` has already called ``setup()``, so the engine is
        detected by the time gates run.
        """
        from misterdev.environments.container_env import (
            ContainerEnvironmentManager,
        )

        env = project.env_manager
        if isinstance(env, ContainerEnvironmentManager):
            return env.engine()
        return None

    def _project_file_map(self, project: Project) -> str:
        """The project's real file+symbol outline, for grounding decomposition.

        Best-effort: builds the symbol graph (idempotent) and returns its project
        outline, or "" if topography is unavailable or errors — the decomposer
        then falls back to cautious path inference rather than failing.
        """
        cached = getattr(project, "_file_map_cache", None)
        if cached is not None:
            return cached
        topo = getattr(project, "topography", None)
        if topo is None:
            return ""
        try:
            topo.initialize()
            result = topo.get_project_outline()
            project._file_map_cache = result
            return result
        except Exception as e:
            logger.warning(f"File map unavailable for decomposition (non-fatal): {e}")
            return ""

    @staticmethod
    def _resolve_claim_file(root: Path, label: str, file_map: str) -> Optional[Path]:
        """Best-effort map a claim label to a real file for evidence.

        A label that is itself a path wins. Otherwise its DISTINCTIVE identifier
        tokens (length >= 5, or CamelCase) are matched as WHOLE WORDS against the
        file+symbol map, longest first, and the first file that mentions one is
        used. The word-boundary + distinctiveness rules stop a generic token like
        "backend" from matching an unrelated ``backend_registry.py``. Returns None
        when nothing resolves — the caller then leaves the claim unverified rather
        than judging it against the wrong file.
        """
        if label:
            direct = root / label
            if direct.is_file():
                return direct
        tokens = sorted(
            {
                t
                for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", label or "")
                if len(t) >= 5 or any(c.isupper() for c in t)
            },
            key=len,
            reverse=True,
        )
        for token in tokens:
            pattern = re.compile(rf"\b{re.escape(token)}\b")
            for line in file_map.splitlines():
                if pattern.search(line):
                    cand = root / line.split(":", 1)[0].strip()
                    if cand.is_file():
                        return cand
        return None

    def _verify_completeness_claims(
        self, project: Project, assessment: ProjectAssessment, report: BuildReport
    ) -> None:
        """Drop completeness claims a second component refutes against the source.

        The analyzer flags "incomplete"/"stub" items from a lossy overview, so it
        can mislabel deliberate design (graceful-degradation, platform no-ops,
        parity shims) as work. Before the spec and tasks are built, recheck each
        claim against the REAL file plus the verified build/test state with an
        independent verifier and prune only the ones it refutes WITH evidence;
        unsure / skip / timeout keeps the claim, so genuine work is never lost.
        No-op when disabled or when no LLM verifier is available.
        """
        if not get_setting(project.config, "orchestrator", "verify_claims"):
            return
        feats = assessment.features
        if not feats.incomplete and not feats.stubs:
            logger.info(
                "Completeness-claim verifier: no incomplete/stub claims flagged; "
                "nothing to verify."
            )
            return

        from misterdev.analyzers.project_analyzer import (
            _health_ground_truth,
        )
        from misterdev.core.verification.claim_verifier import (
            Claim,
            verify_claims,
        )

        root = project.path
        health = _health_ground_truth(assessment.health)
        file_map = self._project_file_map(project)

        def read_body(path: Optional[Path]) -> str:
            if path is None or not path.is_file():
                return ""
            try:
                return path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return ""

        # Build a claim ONLY when the claimed file's real source is in hand. Without
        # it the verifier would see just the "build passes" line — which the prompt
        # treats as grounds to refute — and could drop a genuine claim on no
        # evidence, so an unresolved claim is KEPT, unverified. `health` goes FIRST
        # so it survives the verifier's evidence truncation. `origin` is the
        # assessment object/string, so pruning is by identity and a shared or empty
        # label can't drop the wrong claim.
        entries = []  # (Claim, kind, origin)
        unverified = 0
        for fi in feats.incomplete:
            path = self._resolve_claim_file(root, fi.name, file_map)
            body = read_body(path)
            if not body:
                unverified += 1
                continue
            try:
                rel = path.relative_to(root)
            except ValueError:
                rel = path
            entries.append(
                (
                    Claim(
                        kind="incomplete",
                        label=fi.name,
                        description=f"{fi.description} (source file: {rel})",
                        evidence=f"{health}\n\n{body}",
                    ),
                    "incomplete",
                    fi,
                )
            )
        for sp in feats.stubs:
            if not isinstance(sp, str):
                continue
            body = read_body(root / sp)
            if not body:
                unverified += 1
                continue
            entries.append(
                (
                    Claim(
                        kind="stub",
                        label=sp,
                        description=f"flagged as a stub file (source file: {sp})",
                        evidence=f"{health}\n\n{body}",
                    ),
                    "stub",
                    sp,
                )
            )

        if not entries:
            logger.info(
                "Completeness-claim verifier: no claims with readable source to "
                f"verify ({unverified} kept unverified)."
            )
            return

        judge_model = (project.config.get("judge") or {}).get("model")
        timeout = get_setting(project.config, "orchestrator", "verify_claims_timeout")
        logger.info(
            f"Verifying {len(entries)} completeness claim(s) against the real source..."
        )
        try:
            verdicts = verify_claims(
                [claim for claim, _, _ in entries],
                llm_client=project.llm_client,
                model=judge_model,
                timeout=timeout,
            )
        except Exception as e:  # the gate must never crash a build
            logger.warning(f"Completeness-claim verification failed (non-fatal): {e}")
            report.degraded_subsystems.append(f"Claim verifier: {e}")
            return

        drop_incomplete: set[int] = set()
        drop_stubs: set[str] = set()
        for (claim, kind, origin), v in zip(entries, verdicts):
            if not v.refuted:
                continue
            if kind == "incomplete":
                drop_incomplete.add(id(origin))
            else:
                drop_stubs.add(origin)
            msg = f"Dropped phantom completeness claim '{claim.label}': {v.reason}"
            logger.info(msg)
            report.key_decisions.append(msg)

        dropped = len(drop_incomplete) + len(drop_stubs)
        logger.info(
            f"Completeness-claim verification: {len(entries) - dropped} kept, "
            f"{dropped} dropped ({unverified} unverified)."
        )
        if dropped:
            feats.incomplete = [
                fi for fi in feats.incomplete if id(fi) not in drop_incomplete
            ]
            feats.stubs = [sp for sp in feats.stubs if sp not in drop_stubs]

    def _resolve_targets(self, project: Project) -> list[dict]:
        """Explicit ``targets`` if declared, else auto-discovered when enabled.

        Discovered targets are written back into ``project.config['targets']`` so
        the executor's per-task routing and per-target validation see them too.
        """
        explicit = project.config.get("targets") or []
        if explicit:
            return explicit
        if not get_setting(project.config, "orchestrator", "auto_targets"):
            return []
        from misterdev.core.planning.targets import discover_targets

        discovered = discover_targets(str(project.path))
        if discovered:
            names = ", ".join(t["name"] for t in discovered)
            logger.info(
                f"Auto-discovered {len(discovered)} polyglot target(s): {names}"
            )
            project.config["targets"] = discovered
        return discovered

    def _validate_targets(
        self, project: Project, env_activate: Optional[str]
    ) -> list[dict]:
        """Validate each declared target with its OWN toolchain, vs its baseline.

        Closes the multi-target gap where the end-of-run GateKeeper only ran the
        top-level commands. Crucially this compares against each target's baseline
        (measured before the run, stored on ``project.target_baselines``), so a
        target that was ALREADY broken (e.g. a frontend with pre-existing errors)
        is not counted as a failure for a run that never touched it — only a
        genuine REGRESSION fails. Returns [] when no targets are declared, so
        single-target builds are unaffected.
        """
        targets = project.config.get("targets") or []
        if not targets:
            return []
        # getattr seam lets tests inject a fake runner; prod creates a real one.
        executor = getattr(self, "_validate_executor", None) or MarkdownPlanExecutor()
        target_baselines = getattr(project, "target_baselines", {}) or {}
        build_to = get_setting(project.config, "build", "build_timeout")
        test_to = get_setting(project.config, "build", "test_timeout")
        results: list[dict] = []
        for t in targets:
            gate_cmd = t.get("test_command") or t.get("build_command")
            if not gate_cmd:
                continue
            name = t.get("name") or t.get("path") or "?"
            tp = (t.get("path") or "").strip("/")
            run_dir = project.path / tp if tp else project.path
            timeout = test_to if t.get("test_command") else build_to
            after = self._suite_failures(
                project, executor, gate_cmd, timeout, cwd=run_dir
            )
            baseline = target_baselines.get(name)
            regressed = self._target_regressed(after, baseline)
            ok = not regressed
            detail = "ok" if ok else f"regressed (baseline={baseline}, after={after})"
            # Behavioral verification (opt-in): a frontend target may declare a
            # `web`/`vision` block to verify it actually RENDERS/works, not just
            # type-checks. Only run when the build/test gate is already clean.
            if ok and (t.get("web") or t.get("vision")):
                ok, rt_detail = self._run_target_runtime_gates(project, t, run_dir)
                if not ok:
                    detail = rt_detail
            results.append({"name": name, "ok": ok, "detail": detail})
        return results

    def _run_target_runtime_gates(
        self, project: Project, target: dict, run_dir
    ) -> tuple[bool, str]:
        """Run a target's opt-in web/vision behavioral gates in its directory.

        Mirrors the GateKeeper's G4.7/G4.8 but scoped to a sub-project: the web
        gate renders + screenshots, the vision gate judges that screenshot. Both
        are best-effort and timeout-bounded — only a RED (a real failed check)
        fails the target; a SKIP (no browser/model/config) passes.
        """
        evidence = None
        web_cfg = target.get("web")
        if web_cfg:
            from misterdev.core.verification.web_verify import (
                run_web_gate,
            )

            web = run_web_gate(run_dir, web_cfg)
            evidence = getattr(web, "evidence", None)
            if web.status == "red":
                return False, f"web verify failed ({web.reason or 'no detail'})"
        vision_cfg = target.get("vision")
        if vision_cfg:
            from misterdev.core.verification.vision_verify import (
                run_vision_gate,
            )

            vc = dict(vision_cfg)
            if not vc.get("capture") and evidence:
                vc["capture"] = evidence
            vision = run_vision_gate(
                run_dir, vc or None, llm_client=getattr(project, "llm_client", None)
            )
            if vision.status == "red":
                return False, f"vision verify failed ({vision.reason or 'no detail'})"
        return True, "ok"

    def _analyze(self, project: Project, env_activate: Optional[str]):
        """Phase 1 analysis with config-driven commands and timeouts.

        Shared by build() and interactive_plan() so the analyzer's parameters
        (and any future config wiring) live in exactly one place.
        """
        # Build the project's symbol graph ONCE via its TopographyEngine and feed
        # the outline to the analyzer, instead of letting the source overview parse
        # a second throwaway graph. The engine's initialize() is idempotent, so the
        # later decomposition/file-map calls reuse this same graph.
        project_outline = self._project_file_map(project) or None
        return analyze_project(
            project.path,
            project.llm_client,
            build_command=project.config.get("build_command"),
            test_command=project.config.get("test_command"),
            lint_command=project.config.get("lint_command"),
            env_activate=env_activate,
            build_timeout=get_setting(project.config, "build", "build_timeout"),
            test_timeout=get_setting(project.config, "build", "test_timeout"),
            lint_timeout=get_setting(project.config, "build", "lint_timeout"),
            parallel=get_setting(project.config, "build", "parallel_analysis"),
            project_outline=project_outline,
        )

    def _get_or_register(self, project_path: str | Path) -> Optional[Project]:
        project = self.registry.get_project(project_path)
        if not project:
            try:
                project = self.registry.register_project(project_path)
            except Exception as e:
                logger.error(f"Failed to register project at {project_path}: {e}")
                return None
        return project
