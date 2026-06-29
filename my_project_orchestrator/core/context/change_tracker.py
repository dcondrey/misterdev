"""Diff-based change tracking across tasks.

Records what each completed task changed (files, additions, deletions).
Before a downstream task executes, injects relevant recent changes so
the LLM knows the current state of files it depends on.

Addresses ~20% of later-task failures where the LLM operates on stale
assumptions about code that was modified by earlier tasks.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List

from my_project_orchestrator.logging_setup import setup_logger
from my_project_orchestrator.utils.file_utils import (
    atomic_write,
    orchestrator_state_file,
)

logger = setup_logger(__name__)


class TaskChange:
    def __init__(
        self,
        task_id: str,
        files: List[str],
        diff_summary: str,
        additions: int,
        deletions: int,
    ):
        self.task_id = task_id
        self.files = files
        self.diff_summary = diff_summary
        self.additions = additions
        self.deletions = deletions

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "files": self.files,
            "diff_summary": self.diff_summary,
            "additions": self.additions,
            "deletions": self.deletions,
        }

    @staticmethod
    def from_dict(data: dict) -> "TaskChange":
        return TaskChange(
            task_id=data["task_id"],
            files=data.get("files", []),
            diff_summary=data.get("diff_summary", ""),
            additions=data.get("additions", 0),
            deletions=data.get("deletions", 0),
        )


class ChangeTracker:
    """Tracks what each task changed for downstream context injection."""

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self._file = orchestrator_state_file(project_path, "changes.json")
        self.changes: Dict[str, TaskChange] = {}
        self._load()

    def _load(self):
        if self._file.exists():
            try:
                data = json.loads(self._file.read_text(encoding="utf-8"))
                for task_id, entry in data.items():
                    self.changes[task_id] = TaskChange.from_dict(entry)
            except (json.JSONDecodeError, OSError, KeyError):
                self.changes = {}

    def _save(self):
        data = {tid: tc.to_dict() for tid, tc in self.changes.items()}
        atomic_write(self._file, json.dumps(data, indent=2))

    def record_task_changes(
        self, task_id: str, modified_files: List[str]
    ) -> TaskChange:
        """Record what a completed task changed. Call after task commit."""
        diff_summary = self._get_task_diff(task_id, modified_files)
        additions, deletions = self._count_diff_stats(diff_summary)

        change = TaskChange(
            task_id=task_id,
            files=modified_files,
            diff_summary=diff_summary[:2000],  # cap stored diff
            additions=additions,
            deletions=deletions,
        )
        self.changes[task_id] = change
        self._save()
        logger.info(
            f"Tracked changes for {task_id}: {len(modified_files)} files, +{additions}/-{deletions}"
        )
        return change

    def get_recent_changes_for_files(
        self, target_files: List[str], max_entries: int = 5
    ) -> str:
        """Get recent changes that touched any of the target files."""
        relevant = []
        for change in reversed(list(self.changes.values())):
            overlap = set(change.files) & set(target_files)
            if overlap:
                relevant.append((change, overlap))
            if len(relevant) >= max_entries:
                break

        if not relevant:
            return ""

        lines = ["## Recent Changes to Related Files"]
        for change, overlap_files in relevant:
            lines.append(f"\n### {change.task_id} (touched {', '.join(overlap_files)})")
            lines.append(f"+{change.additions}/-{change.deletions} lines")
            if change.diff_summary:
                # Show only the parts relevant to overlapping files
                filtered = self._filter_diff_for_files(
                    change.diff_summary, overlap_files
                )
                if filtered:
                    lines.append(f"```diff\n{filtered}\n```")

        return "\n".join(lines)

    def _get_task_diff(self, task_id: str, files: List[str]) -> str:
        """Return the diff produced by a single task.

        Each task commits with a message ``task(<id>): ...``. Looking the
        commit up by ID means every task is attributed its own diff regardless
        of how many other tasks committed before this is recorded (a plain
        ``HEAD~1 HEAD`` would give every task in a batch the same final diff).
        Falls back to a file-scoped working-tree diff when the commit can't be
        found (e.g. file-snapshot mode in a non-git repo).
        """
        sha = self._find_task_commit(task_id)
        if sha:
            diff = self._run_git(["diff", f"{sha}~1", sha, "--unified=3", "--", *files])
            if diff:
                return diff
        # Fallback for tasks not committed under the task(<id>): convention:
        # the most recent commit, scoped to the task's files when known.
        scope = ["--", *files] if files else []
        diff = self._run_git(["diff", "HEAD~1", "HEAD", "--unified=3", *scope])
        if diff:
            return diff
        # Last resort: uncommitted working-tree changes for the task's files.
        if files:
            return self._run_git(["diff", "HEAD", "--unified=3", "--", *files])
        return ""

    def _find_task_commit(self, task_id: str) -> str:
        """Find the SHA of the commit recording this task, if any."""
        out = self._run_git(
            [
                "log",
                "--all",
                "-n",
                "1",
                "--format=%H",
                "--fixed-strings",
                f"--grep=task({task_id}):",
            ]
        )
        lines = out.splitlines()
        return lines[0].strip() if lines else ""

    def _run_git(self, args: List[str]) -> str:
        """Run a git command (list form, no shell) and return stdout, or ''."""
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return proc.stdout.strip() if proc.returncode == 0 else ""
        except Exception as e:
            logger.debug(f"Could not run git {args[0] if args else ''}: {e}")
            return ""

    def _count_diff_stats(self, diff: str) -> tuple:
        additions = 0
        deletions = 0
        for line in diff.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                additions += 1
            elif line.startswith("-") and not line.startswith("---"):
                deletions += 1
        return additions, deletions

    def _filter_diff_for_files(
        self, diff: str, target_files: set, max_lines: int = 30
    ) -> str:
        """Extract diff hunks for specific files only."""
        lines = diff.splitlines()
        output = []
        in_relevant_file = False
        line_count = 0

        for line in lines:
            if line.startswith("diff --git"):
                in_relevant_file = any(f in line for f in target_files)
            if in_relevant_file:
                output.append(line)
                line_count += 1
                if line_count >= max_lines:
                    output.append("... (truncated)")
                    break

        return "\n".join(output)
