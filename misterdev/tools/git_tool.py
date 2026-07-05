import shlex
from typing import Any, Tuple

from misterdev.tools.command import CommandTool
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)


class GitTool(CommandTool):
    """Tool for common Git operations including worktrees and branching."""

    def execute(self, project: Any, action: str, **kwargs) -> Tuple[bool, str]:
        if action == "status":
            return self.status(project)
        elif action == "add":
            return self.add(project, kwargs.get("files", "."))
        elif action == "commit":
            return self.commit(
                project, kwargs.get("message", "Auto-commit by Project Orchestrator")
            )
        elif action == "diff":
            return self.diff(project)
        elif action == "worktree_add":
            return self.worktree_add(project, kwargs.get("path"), kwargs.get("branch"))
        elif action == "worktree_remove":
            return self.worktree_remove(project, kwargs.get("path"))
        elif action == "branch_create":
            return self.branch_create(project, kwargs.get("branch"))
        elif action == "branch_delete":
            return self.branch_delete(project, kwargs.get("branch"))
        elif action == "merge":
            return self.merge(project, kwargs.get("branch"))
        elif action == "checkout":
            return self.checkout(project, kwargs.get("branch"))
        else:
            return super().execute(project, command=f"git {shlex.quote(action)}")

    def worktree_add(
        self, project: Any, path: str, branch: str, new_branch: bool = False
    ) -> Tuple[bool, str]:
        logger.info(
            f"Creating git worktree at {path} (branch: {branch}, new={new_branch})"
        )
        flag = "-b " if new_branch else ""
        return super().execute(
            project,
            command=f"git worktree add {shlex.quote(path)} {flag}{shlex.quote(branch)}",
        )

    def worktree_remove(self, project: Any, path: str) -> Tuple[bool, str]:
        logger.info(f"Removing git worktree at {path}")
        return super().execute(
            project, command=f"git worktree remove --force {shlex.quote(path)}"
        )

    def merge_worktree(self, project: Any, branch: str) -> Tuple[bool, str]:
        """Merge a worktree's branch into the current branch, then delete it."""
        success, out = super().execute(
            project, command=f"git merge --no-ff {shlex.quote(branch)} --no-edit"
        )
        if success:
            super().execute(project, command=f"git branch -d {shlex.quote(branch)}")
        return success, out

    def branch_create(self, project: Any, branch: str) -> Tuple[bool, str]:
        return super().execute(
            project, command=f"git checkout -b {shlex.quote(branch)}"
        )

    def branch_delete(self, project: Any, branch: str) -> Tuple[bool, str]:
        return super().execute(project, command=f"git branch -D {shlex.quote(branch)}")

    def merge(self, project: Any, branch: str) -> Tuple[bool, str]:
        return super().execute(project, command=f"git merge {shlex.quote(branch)}")

    def checkout(self, project: Any, branch: str) -> Tuple[bool, str]:
        return super().execute(project, command=f"git checkout {shlex.quote(branch)}")

    def status(self, project: Any) -> Tuple[bool, str]:
        return super().execute(project, command="git status")

    def add(self, project: Any, files: str) -> Tuple[bool, str]:
        return super().execute(project, command=f"git add {shlex.quote(files)}")

    def commit(self, project: Any, message: str) -> Tuple[bool, str]:
        return super().execute(project, command=f"git commit -m {shlex.quote(message)}")

    def diff(self, project: Any) -> Tuple[bool, str]:
        return super().execute(project, command="git diff")
