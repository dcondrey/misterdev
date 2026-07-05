"""Pre-flight devplan validation.

Validates that a devplan is well-formed and executable before any
(money-spending) LLM calls are made: missing context files, dangling
dependency references, uninstalled test-command binaries, duplicate file
targets, and missing titles.
"""

import shutil
from pathlib import Path
from typing import List

from misterdev.core.models import Task
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)


class PreflightIssue:
    """A single problem found while validating a devplan task."""

    def __init__(self, task_id: str, severity: str, message: str):
        self.task_id = task_id
        self.severity = severity  # "error" | "warning"
        self.message = message

    def __repr__(self) -> str:
        return f"[{self.severity.upper()}] {self.task_id}: {self.message}"


class PreflightValidator:
    """Validates devplan tasks before execution to catch issues early."""

    def validate(self, tasks: List[Task], project_path: Path) -> List[PreflightIssue]:
        issues: List[PreflightIssue] = []
        task_ids = {t.id for t in tasks}
        project_path = Path(project_path)

        # Map file -> tasks that modify it (to warn about conflicting targets).
        modifiers: dict[str, list[str]] = {}

        for task in tasks:
            for f in task.context_files:
                if not (project_path / f).exists():
                    issues.append(
                        PreflightIssue(
                            task.id, "warning", f"Context file '{f}' does not exist"
                        )
                    )

            for dep in task.dependencies:
                if dep not in task_ids:
                    issues.append(
                        PreflightIssue(
                            task.id,
                            "error",
                            f"Dependency '{dep}' does not match any task",
                        )
                    )

            test_cmd = task.processor_data.get("test_command")
            if test_cmd:
                binary = test_cmd.split()[0]
                if not shutil.which(binary):
                    issues.append(
                        PreflightIssue(
                            task.id,
                            "warning",
                            f"Test command binary '{binary}' not found in PATH",
                        )
                    )

            if not task.title:
                issues.append(PreflightIssue(task.id, "warning", "Task has no title"))

            for f in task.files_to_modify:
                modifiers.setdefault(f, []).append(task.id)

        for file_path, tids in modifiers.items():
            independent = [t for t in tasks if t.id in tids and not t.dependencies]
            if len(tids) > 1 and len(independent) > 1:
                issues.append(
                    PreflightIssue(
                        independent[1],
                        "warning",
                        f"File '{file_path}' modified by multiple independent tasks "
                        f"({', '.join(tids)}); they may conflict if run in the same wave",
                    )
                )

        return issues

    @staticmethod
    def has_errors(issues: List[PreflightIssue]) -> bool:
        return any(i.severity == "error" for i in issues)
