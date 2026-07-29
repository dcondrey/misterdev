"""Parallel / worktree task execution, extracted from the orchestrator.

``ProjectOrchestrator`` mixes this in: every method uses only ``self`` (all of its
collaborators live here or on the coordinator) plus module-level helpers, so the
split is behavior-preserving — method resolution and call sites are unchanged.
The unit owns how a wave of ready tasks runs concurrently: worktree isolation
(prime by CoW clone or install, sanity-probe, conflict-free sub-wave batching),
serial merge-back with a post-merge base gate, and the shared-tree fallback.
"""

import concurrent.futures
from typing import Optional

from misterdev.agent_helpers import (
    _WorktreeProjectView,
    worktree_healthcheck_command,
    worktree_setup_command,
)
from misterdev.config import get_setting
from misterdev.core.execution.project import Project
from misterdev.core.execution.wave_partition import partition_parallel_safe
from misterdev.core.models import Task
from misterdev.logging_setup import setup_logger
from misterdev.task_executors.markdown_plan_executor import MarkdownPlanExecutor

logger = setup_logger(__name__)


class _BrokenBaseError(RuntimeError):
    """Raised when a post-merge rollback fails and the base branch is left corrupt."""


class ParallelExecutionMixin:
    @staticmethod
    def _task_file_set(task: Task) -> set:
        """Declared files a task will touch (modify + create).

        Tolerates non-list values (e.g. unconfigured mocks): only real lists
        contribute paths, anything else is treated as "unknown / no claim".
        """
        files: set = set()
        for attr in ("files_to_modify", "files_to_create"):
            value = getattr(task, attr, None)
            if isinstance(value, list):
                files.update(str(p) for p in value)
        return files

    @classmethod
    def _partition_disjoint(cls, ready: list[Task]) -> tuple[list, list]:
        """Split tasks into a concurrent-safe group + a serial remainder.

        A task joins the concurrent group only if its declared file set is
        disjoint from every task already in that group; otherwise it is
        deferred to the serial remainder so overlapping writes can't interleave.
        """
        concurrent_group: list = []
        serial_remainder: list = []
        claimed: set = set()
        for task in ready:
            files = cls._task_file_set(task)
            if not files or files & claimed:
                serial_remainder.append(task)
            else:
                concurrent_group.append(task)
                claimed |= files
        return concurrent_group, serial_remainder

    def _execute_parallel(
        self, ready: list[Task], executor: MarkdownPlanExecutor, project: Project
    ) -> list:
        """Execute a batch of independent tasks concurrently.

        In "worktree" mode each task runs in its own git worktree so parallel
        edits can't collide. When the mode is left at its default and the
        project is a git repo, worktree isolation is preferred automatically;
        "shared" must be requested explicitly to opt out. In shared mode only
        tasks with disjoint declared file sets run in the same concurrent batch;
        tasks whose file sets overlap are run serially afterwards.
        """
        mode = get_setting(project.config, "orchestrator", "parallel_mode")
        is_git_repo = (project.path / ".git").exists() is True
        # "auto" (default) isolates via worktrees on a git repo; the value itself
        # carries the intent, so no fragile "was it explicitly set" detection.
        prefer_worktrees = mode == "worktree" or (mode == "auto" and is_git_repo)
        if prefer_worktrees and is_git_repo:
            return self._execute_parallel_worktrees(ready, executor, project)

        concurrent_group, serial_remainder = self._partition_disjoint(ready)
        results = []
        max_workers = get_setting(project.config, "orchestrator", "max_workers")
        if concurrent_group:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(len(concurrent_group), max_workers)
            ) as pool:
                future_to_task = {
                    pool.submit(
                        executor.execute, task, project, use_git_branch=False
                    ): task
                    for task in concurrent_group
                }
                for future in concurrent.futures.as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        results.append((task, result, None))
                    except Exception as e:
                        results.append((task, None, e))

        # Tasks with overlapping file claims run one at a time.
        for task in serial_remainder:
            try:
                result = executor.execute(task, project, use_git_branch=False)
                results.append((task, result, None))
            except Exception as e:
                results.append((task, None, e))
        return results

    def _worktree_setup_command(self, project: Project) -> Optional[str]:
        """The command that primes a fresh worktree's dependencies before gating,
        or None to skip. Delegates to the shared resolver so worktree creation and
        the per-gate infra-reprime helper agree on one command."""
        return worktree_setup_command(project.config, project.path)

    def _worktree_healthcheck_command(self, project: Project) -> Optional[str]:
        """The fast probe that confirms a primed worktree's toolchain resolves, or
        None to skip. Delegates to the shared resolver (auto-detected for node/pnpm
        projects; overridable via ``orchestrator.worktree_healthcheck_command``)."""
        return worktree_healthcheck_command(project.config, project.path)

    def _prepare_task_worktree(
        self,
        project,
        git,
        task,
        wt_root,
        clone_deps,
        health_cmd,
        setup_cmd,
        setup_timeout,
        cmd_tool,
    ):
        """Create + prime one task's worktree. Returns ``(prep, error)`` where
        ``prep`` is ``(task, wt_path, branch)`` on success and ``error`` is set on
        failure (with the worktree already torn down).

        A prep step (clone-prime / install / healthcheck) that RAISES after the
        worktree was created would otherwise leak that worktree — the wave cleanup
        only iterates fully-prepared tasks. Tear it down here so the leak can't
        happen, and surface the task as errored instead of aborting the batch.
        """
        import uuid

        # A run-unique branch name (never a bare ``task/<id>``): a leftover branch
        # from a prior failed run must not collide with this run's ``-b`` create.
        # The unique branch is cut fresh from HEAD, so it carries the latest
        # committed work (including earlier sub-waves).
        run_id = uuid.uuid4().hex[:6]
        branch = f"task/{task.id}-{run_id}"
        wt_path = wt_root / f"{task.id}-{run_id}"
        ok, out = git.worktree_add(project, str(wt_path), branch, new_branch=True)
        if not ok:
            logger.error(f"Worktree add failed for {task.id}: {out}")
            return None, RuntimeError(f"worktree add failed: {out}")
        try:
            # Prime deps ONCE here (serially, off the parallel gate path) so the
            # gate tests code, not install speed. Prefer a near-instant CoW clone
            # of the base node_modules; that path self-verifies with the probe.
            primed = False
            if clone_deps:
                primed = self._prime_worktree_by_clone(
                    project, task, wt_path, health_cmd, setup_timeout, cmd_tool
                )
            # Install fallback: clone unavailable/declined or it failed the probe.
            # Best-effort — the gate's own implicit install is the backstop.
            if not primed and cmd_tool is not None and setup_cmd:
                sok, sout = cmd_tool.execute(
                    project, setup_cmd, cwd=str(wt_path), timeout=setup_timeout
                )
                if not sok:
                    logger.warning(
                        f"Worktree dep prep failed for {task.id} "
                        f"(gate will fall back to its own install): {sout[-200:]}"
                    )
            # A clone already passed the sanity probe; only the install path needs
            # the re-prime-once healthcheck.
            if not primed:
                self._worktree_healthcheck(
                    project,
                    task,
                    wt_path,
                    setup_cmd,
                    health_cmd,
                    setup_timeout,
                    cmd_tool,
                )
            return (task, wt_path, branch), None
        except Exception as e:
            logger.error(
                f"Worktree prep raised for {task.id}; tearing down to avoid a leak: {e}"
            )
            git.worktree_remove(project, str(wt_path))
            git.branch_delete(project, branch)
            return None, e

    def _worktree_healthcheck(
        self, project, task, wt_path, setup_cmd, health_cmd, timeout, cmd_tool
    ) -> None:
        """Confirm the primed worktree's toolchain resolves; heal once or flag it.

        Runs the cheap probe right after priming. On failure, a broken/partial
        install is the likely cause, so re-prime the deps ONCE and re-probe. If it
        still fails, log clearly that THIS WORKTREE's environment is unhealthy —
        not the task's code — so a downstream gate failure here is read as an
        environment fault, not attributed to the task. Best-effort: never raises
        and never drops the task; the gate's own infra self-heal is the backstop.
        """
        if not health_cmd or cmd_tool is None:
            return
        ok, out = cmd_tool.execute(
            project, health_cmd, cwd=str(wt_path), timeout=timeout
        )
        if ok:
            return
        logger.warning(
            f"Worktree health probe failed for {task.id} ({health_cmd!r}); "
            f"re-priming deps once and re-probing: {out[-200:]}"
        )
        if setup_cmd:
            cmd_tool.execute(project, setup_cmd, cwd=str(wt_path), timeout=timeout)
        ok, out = cmd_tool.execute(
            project, health_cmd, cwd=str(wt_path), timeout=timeout
        )
        if not ok:
            logger.error(
                f"Worktree ENVIRONMENT unhealthy for {task.id} after re-prime "
                f"({health_cmd!r}): the toolchain does not resolve in this "
                f"worktree. Downstream gate failures here are an environment "
                f"fault, NOT the task's code: {out[-200:]}"
            )

    def _prime_worktree_by_clone(
        self, project, task, wt_path, health_cmd, timeout, cmd_tool
    ) -> bool:
        """Prime a worktree by copy-on-write cloning the base node_modules.

        Returns True only when the clone succeeded AND the cloned toolchain passes
        the P3 sanity probe (so a cloned-but-broken tree falls back to install).
        Requires a probe command (``health_cmd``) to verify with; without one we
        cannot confirm the clone, so we decline and let the install path run.
        """
        from misterdev.core.execution.dep_clone import clone_dependencies

        if not health_cmd or cmd_tool is None:
            return False
        cloned, dirs = clone_dependencies(project.path, wt_path)
        if not cloned:
            return False
        ok, out = cmd_tool.execute(
            project, health_cmd, cwd=str(wt_path), timeout=timeout
        )
        if not ok:
            logger.info(
                f"Cloned deps for {task.id} failed the sanity probe "
                f"({health_cmd!r}); falling back to install: {out[-200:]}"
            )
            return False
        logger.info(
            f"Primed {task.id} by cloning {len(dirs)} node_modules dir(s) from the "
            "base checkout (copy-on-write, no install)."
        )
        return True

    def _post_merge_healthcheck(
        self,
        project: Project,
        executor: MarkdownPlanExecutor,
        git,
        task: Task,
        timeout: int,
    ) -> bool:
        """Gate the base branch after a task's merge; roll the merge back if broken.

        Runs the merged task's OWNING-target gate (typecheck/test, resolved via the
        same routing the executor uses) on the base checkout. Returns True when the
        base is healthy (or there is nothing to gate). On a real (non-infra)
        failure the merge broke the base, so ``reset --hard HEAD^`` removes the
        merge commit (it is the ``--no-ff`` tip) and returns False — the caller
        then treats the task as unfinished. A transient/infra failure is NOT rolled
        back: it is an environment fault, not a code break, so the merge stands.
        """
        from misterdev.core.planning.targets import select_target, target_commands
        from misterdev.core.execution.infra import infra_failure

        files = list(task.files_to_modify) + list(task.files_to_create)
        targets = project.config.get("targets") or []
        tgt = select_target(targets, files)
        cmds = target_commands(tgt, project.config)
        # Prefer the cheapest reliable signal (typecheck) so a per-merge gate is
        # fast; fall back to test then build when the target declares no typecheck.
        gate_cmd = (
            cmds["typecheck_command"] or cmds["test_command"] or cmds["build_command"]
        )
        if not gate_cmd:
            return True
        tp = (tgt.get("path") or "").strip("/") if tgt else ""
        run_dir = project.path / tp if tp else project.path
        ok, out = executor._run_command(project, gate_cmd, timeout=timeout, cwd=run_dir)
        if ok:
            return True
        infra = infra_failure(out)
        if infra:
            logger.warning(
                f"Post-merge gate for {task.id} failed on an environment fault "
                f"({infra}), not the code; leaving the merge in place: {out[-200:]}"
            )
            return True
        logger.error(
            f"Post-merge gate for {task.id} FAILED on the base branch "
            f"({gate_cmd!r}): the merge broke the base. Rolling it back and "
            f"re-queuing the task: {out[-200:]}"
        )
        rok, rout = git.reset_hard(project, "HEAD^")
        if not rok:
            raise _BrokenBaseError(
                f"Failed to roll back {task.id}'s merge; base branch is broken: "
                f"{rout[-200:]}"
            )
        return False

    def _execute_parallel_worktrees(
        self, ready: list[Task], executor: MarkdownPlanExecutor, project: Project
    ) -> list:
        """Run each task in an isolated git worktree, then merge successes back.

        Worktrees are created and merged serially (git's index/worktree metadata
        is not concurrency-safe); only the task bodies run in parallel. Each fresh
        worktree's dependencies are primed once at creation (see
        ``_worktree_setup_command``) so the parallel gates test code, not install
        speed.
        """
        from misterdev.tools.command import CommandTool
        from misterdev.tools.git_tool import GitTool

        git = GitTool({})
        wt_root = project.path / ".orchestrator" / "worktrees"
        wt_root.mkdir(parents=True, exist_ok=True)
        # Drop metadata from any worktree a prior run left dangling before we add new ones.
        git.worktree_prune(project)
        setup_cmd = self._worktree_setup_command(project)
        setup_timeout = get_setting(
            project.config, "orchestrator", "worktree_setup_timeout"
        )
        health_cmd = self._worktree_healthcheck_command(project)
        cmd_tool = CommandTool({}) if (setup_cmd or health_cmd) else None
        post_merge_hc = get_setting(
            project.config, "orchestrator", "post_merge_healthcheck"
        )
        gate_timeout = get_setting(project.config, "build", "test_timeout")
        # Prefer a copy-on-write clone of the base node_modules over reinstalling.
        # Resolve FS support ONCE (it is the same for every worktree); a non-CoW
        # filesystem (e.g. HFS+) or a missing probe transparently falls back below.
        from misterdev.core.execution.dep_clone import clone_supported

        clone_deps = (
            get_setting(project.config, "orchestrator", "worktree_clone_deps")
            and bool(setup_cmd)
            and bool(health_cmd)
            and clone_supported(project.path)
        )
        results: list = []
        max_workers = get_setting(project.config, "orchestrator", "max_workers")

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

        # Split the wave so tasks that DECLARE a shared file run in different
        # sub-waves (serially): parallel worktrees editing the same file would
        # race and clobber or conflict on merge. Disjoint tasks stay in one batch.
        # Sub-waves run in order, so a later batch's worktrees are cut from HEAD
        # AFTER the earlier batch merged — it builds on that work, not around it.
        if get_setting(project.config, "orchestrator", "serialize_conflicting_tasks"):
            batches = partition_parallel_safe(
                [(t, self._task_file_set(t)) for t in ready]
            )
        else:
            batches = [list(ready)]
        if len(batches) > 1:
            logger.info(
                f"Wave split into {len(batches)} conflict-free sub-wave(s) by "
                f"declared file overlap ({[len(b) for b in batches]} task(s) each)."
            )

        for batch in batches:
            prepared: list = []
            for task in batch:
                prep, err = self._prepare_task_worktree(
                    project,
                    git,
                    task,
                    wt_root,
                    clone_deps,
                    health_cmd,
                    setup_cmd,
                    setup_timeout,
                    cmd_tool,
                )
                if prep is not None:
                    prepared.append(prep)
                elif err is not None:
                    results.append((task, None, err))

            raw = []
            if prepared:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(len(prepared), max_workers)
                ) as pool:
                    futures = [pool.submit(run_one, item) for item in prepared]
                    raw = [f.result() for f in concurrent.futures.as_completed(futures)]

            base_broken = False
            for task, result, error, wt_path, branch in raw:
                # Remove the worktree BEFORE merging/deleting the branch: git refuses
                # to delete a branch still checked out in a worktree, which otherwise
                # leaks even merged branches. The task's commits live on the branch
                # ref, so the merge below still sees them after the dir is gone.
                ok, msg = git.worktree_remove(project, str(wt_path))
                if not ok:
                    logger.warning(f"Failed to remove worktree {wt_path}: {msg}")
                merged = False
                if base_broken:
                    error = RuntimeError(
                        "base branch broken from prior rollback failure; skipping merge"
                    )
                    result = None
                elif (
                    result is not None
                    and getattr(result, "status", None) == "completed"
                ):
                    merged, mout = git.merge_worktree(project, branch)
                    if not merged:
                        # merge_worktree already aborted the conflicted merge, so the
                        # base is clean; re-queue the task (unfinished) rather than
                        # force-merging over another task's shared-file change.
                        logger.error(
                            f"Worktree merge conflicted for {task.id}; aborted and "
                            f"re-queuing (not force-merged): {mout[-200:]}"
                        )
                        error, result = RuntimeError(f"merge conflict: {mout}"), None
                    else:
                        try:
                            if post_merge_hc and not self._post_merge_healthcheck(
                                project, executor, git, task, gate_timeout
                            ):
                                # The merge broke the base branch and was rolled back; treat
                                # the task as unfinished (not completed) so it is retried and
                                # not recorded as done. The branch was already deleted by the
                                # (successful) merge, so no extra cleanup is needed.
                                error, result = (
                                    RuntimeError(
                                        "post-merge health gate failed; base merge rolled back"
                                    ),
                                    None,
                                )
                        except _BrokenBaseError as e:
                            logger.error(str(e))
                            base_broken = True
                            error, result = RuntimeError(str(e)), None
                # A successful merge already deleted the branch; drop any un-merged
                # one so no throwaway branch accumulates or collides with a later run.
                if not merged:
                    git.branch_delete(project, branch)
                results.append((task, result, error))
        return results
