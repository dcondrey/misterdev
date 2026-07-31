"""ExecutionLoopMixin — the wave-based task execution loop for ProjectOrchestrator.

Extracted from agent.py. _execute_tasks calls self.* methods that are each
resolved through the MRO at runtime (IntegrationGateMixin, WaveMixin,
InteractiveMixin, ParallelExecutionMixin, ReportingMixin). No back-refs to
instance attributes defined in ProjectOrchestrator.__init__.
"""

from pathlib import Path
from typing import Callable, Optional

from misterdev.agent_helpers import _budget_exhausted, _combine_commands
from misterdev.config import get_setting
from misterdev.core.context.change_tracker import ChangeTracker
from misterdev.core.context.contracts import ContractRegistry
from misterdev.core.context.scratchpad import Scratchpad
from misterdev.core.execution.progress import ProgressTracker, compute_task_hash
from misterdev.core.execution.project import Project
from misterdev.core.models import Task
from misterdev.core.modes import BuildFlags
from misterdev.core.planning.sovereign import RealTimeAligner, StrategyOptimizer
from misterdev.core.reporting.report import BuildReport
from misterdev.llm.client import BudgetExceededError
from misterdev.logging_setup import setup_logger
from misterdev.task_executors.markdown_plan_executor import MarkdownPlanExecutor

logger = setup_logger(__name__)

MAX_CONSECUTIVE_FAILURES = 3


class ExecutionLoopMixin:
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
        wave_tuning = WaveTuning(int(adaptive_base["workers"] or 1), 1.0)
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
