"""Git branch-per-task operations and regression bisection."""

import shlex
from pathlib import Path
from typing import Dict, List, Optional

from misterdev.core.models import Task
from misterdev.core.execution.project import Project
from misterdev.core.verification.validator import _run_cmd

from .helpers import logger, _bisect_first_failing


class GitMixin:
    # ----------------------------------------------------------------
    # Git branch-per-task operations
    # ----------------------------------------------------------------

    def _is_git_repo(self, project: Project) -> bool:
        return (project.path / ".git").exists()

    # ----------------------------------------------------------------
    # Regression bisection (post-build gate failure)
    # ----------------------------------------------------------------

    def find_task_commit(self, project: Project, task_id: str) -> Optional[str]:
        """SHA of the commit recording a task (message 'task(<id>):'), or None."""
        ok, out = self._git(
            project,
            f"git log --all -n 1 --format=%H --fixed-strings --grep={shlex.quote(f'task({task_id}):')}",
        )
        sha = out.strip().splitlines()[0] if ok and out.strip() else ""
        return sha or None

    def bisect_regression(
        self,
        project: Project,
        task_commits: List,
        test_command: str,
        timeout: int = 180,
    ) -> Optional[str]:
        """Find the earliest task commit whose checkout fails the test command.

        task_commits is [(task_id, sha)] ordered oldest->newest. Returns the
        culprit task_id, or None if no checked-out commit actually fails (so a
        flaky/non-task regression isn't misattributed). Restores HEAD after.
        """
        if not task_commits:
            return None
        # Restore to the BRANCH, not a bare SHA: checking out a SHA detaches
        # HEAD, and a mid-build gate that left HEAD detached would make every
        # subsequent task branch from / merge into a detached head instead of
        # the working branch, silently diverging the build from the branch ref.
        okb, branch = self._git(project, "git rev-parse --abbrev-ref HEAD")
        branch = branch.strip() if okb else ""
        if branch and branch != "HEAD":
            restore = branch
        else:
            ok, head = self._git(project, "git rev-parse HEAD")
            restore = head.strip() if ok else None

        def passes_at(i: int) -> bool:
            self._git(project, f"git checkout {shlex.quote(task_commits[i][1])}")
            success, _ = self._run_command(project, test_command, timeout=timeout)
            return success

        try:
            idx = _bisect_first_failing(len(task_commits), passes_at)
            culprit = None if passes_at(idx) else task_commits[idx][0]
        finally:
            if restore:
                self._git(project, f"git checkout {shlex.quote(restore)}")
        return culprit

    def revert_task_commit(self, project: Project, sha: str) -> bool:
        """Revert a task's commit, leaving an explicit revert commit.

        Aborts cleanly on conflict so a failed revert never leaves the working
        tree in a half-reverted, conflict-marked state that would break later
        suite runs and checkouts.
        """
        ok, _ = self._git(project, f"git revert --no-edit {shlex.quote(sha)}")
        if not ok:
            self._git(project, "git revert --abort")
        return ok

    def _get_current_branch(self, project: Project) -> Optional[str]:
        ok, output = self._git(project, "git rev-parse --abbrev-ref HEAD")
        return output.strip() if ok else None

    def _create_task_branch(self, project: Project, branch_name: str) -> bool:
        ok, _ = self._git(project, f"git checkout -b {shlex.quote(branch_name)}")
        if ok:
            logger.info(f"Created task branch: {branch_name}")
        return ok

    def _commit_task(
        self,
        project: Project,
        branch_name: Optional[str],
        base_branch: Optional[str],
        task: Task,
        files: Optional[List[str]] = None,
    ):
        """Commit the task's own changes and merge the task branch back to base.

        Stages only the named files, never `git add -A`: a blanket add sweeps
        unrelated untracked files (other uncommitted user work, or files carried
        onto the task branch) into the commit, which then get destroyed if the
        task is later reverted. With no files, commit empty rather than sweep.
        """
        msg = f"task({task.id}): {task.title or task.description[:50]}"
        stage = list(files or [])
        # Also commit the task's own source markdown so a status:completed write
        # rides into the merge. Without this the status write is left uncommitted
        # and the NEXT task's `git checkout -- .` (below) wipes it, so a finished
        # devplan showed every task still "pending" (only the last survived).
        source_rel = self._repo_relative(project, getattr(task, "source_ref", None))
        if source_rel:
            stage.append(source_rel)
        if stage:
            quoted = " ".join(shlex.quote(f) for f in stage)
            self._git(project, f"git add -- {quoted}")
        self._git(project, f"git commit -m {shlex.quote(msg)} --allow-empty")

        if branch_name and base_branch:
            # Drop tracked spillover before switching branches: a project-wide
            # formatter (e.g. `ruff format .`) reformats files outside the task,
            # which aren't committed and would otherwise be carried across the
            # checkout and accumulate as a permanently dirty tree.
            self._git(project, "git checkout -- .")
            self._git(project, f"git checkout {shlex.quote(base_branch)}")
            ok, output = self._git(
                project,
                f"git merge --no-ff {shlex.quote(branch_name)} -m {shlex.quote(f'Merge {branch_name}')}",
            )
            if ok:
                self._git(project, f"git branch -d {shlex.quote(branch_name)}")
                logger.info(f"Merged and cleaned up branch: {branch_name}")
            else:
                logger.error(f"Merge failed for {branch_name}: {output}")

    def _repo_relative(self, project: Project, path: Optional[str]) -> Optional[str]:
        """Repo-relative form of a path inside the project, else None.

        Returns None for an empty path or one outside the repo (e.g. a
        decomposed build() task with no backing file), so those are never staged.
        """
        if not path:
            return None
        try:
            return str(Path(path).resolve().relative_to(project.path.resolve()))
        except ValueError:
            return None

    def _untracked_files(self, project: Project) -> set:
        """Set of untracked (non-ignored) paths in the working tree.

        Used to bound revert cleanup: only files that became untracked DURING a
        task are removed, so a user's pre-existing untracked work is never
        touched.
        """
        ok, out = self._git(project, "git status --porcelain --untracked-files=all")
        if not ok:
            return set()
        paths = set()
        for line in out.splitlines():
            if line.startswith("?? "):
                paths.add(line[3:].strip().strip('"'))
        return paths

    def _abort_task(
        self,
        project: Project,
        branch_name: Optional[str],
        base_branch: Optional[str],
        snapshot: Optional[Dict],
        untracked_before: Optional[set] = None,
    ):
        """Revert a failed task: delete branch or restore file snapshots.

        Also removes files the task left UNTRACKED (new files it wrote, whether
        or not committed on the branch) — ``git reset --hard`` reverts tracked
        changes but leaves those orphans behind, which otherwise accumulate in
        the tree (the emathy run left signing.rs/crdt.rs/out/ this way). Only the
        delta against ``untracked_before`` is cleaned, so pre-existing untracked
        files are preserved.
        """
        if branch_name and base_branch:
            self._git(project, "git reset --hard")
            self._git(project, f"git checkout {shlex.quote(base_branch)}")
            self._git(project, f"git branch -D {shlex.quote(branch_name)}")
            logger.info(f"Aborted and deleted branch: {branch_name}")
            self._clean_task_orphans(project, untracked_before)
        elif snapshot is not None:
            self._revert_files(project, snapshot)
            self._clean_task_orphans(project, untracked_before)

    def _clean_task_orphans(
        self, project: Project, untracked_before: Optional[set]
    ) -> None:
        """Remove only files that became untracked during the aborted task."""
        if untracked_before is None:
            return
        new_orphans = self._untracked_files(project) - untracked_before
        if not new_orphans:
            return
        paths = " ".join(shlex.quote(p) for p in sorted(new_orphans))
        self._git(project, f"git clean -fd -- {paths}")
        logger.info(f"Cleaned {len(new_orphans)} orphan file(s) from reverted task.")

    def _git(self, project: Project, command: str) -> tuple:
        ok, output = _run_cmd(command, project.path, None, timeout=30)
        return ok, output.strip()
