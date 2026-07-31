import json
import re
import shlex
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from rich.console import Console
from rich.panel import Panel

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
from misterdev.core.planning.decomposer import topological_sort
from misterdev.core.gitcmd import run_git
from misterdev.core.planning.sovereign import StrategyOptimizer
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
from misterdev.core.execution.pipeline_mixin import PipelineMixin
from misterdev.core.execution.interactive_plan_mixin import InteractivePlanMixin
from misterdev.core.models import Task
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
    PipelineMixin,
    InteractivePlanMixin,
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


    def _get_or_register(self, project_path: str | Path) -> Optional[Project]:
        project = self.registry.get_project(project_path)
        if not project:
            try:
                project = self.registry.register_project(project_path)
            except Exception as e:
                logger.error(f"Failed to register project at {project_path}: {e}")
                return None
        return project
