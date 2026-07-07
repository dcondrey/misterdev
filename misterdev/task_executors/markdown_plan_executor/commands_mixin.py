"""Command execution and file snapshot operations."""

from typing import Dict, List, Optional

from misterdev.core.models import Task
from misterdev.core.execution.project import Project
from misterdev.core.verification.validator import _run_cmd
from misterdev.utils.file_utils import write_file

from .helpers import logger


class CommandsMixin:
    # ----------------------------------------------------------------
    # Command execution and file operations
    # ----------------------------------------------------------------

    def _run_command(
        self, project: Project, command: str, timeout: int = 120, cwd=None
    ) -> tuple:
        # cwd lets a routed multi-target task run its gate in the TARGET's
        # directory (e.g. `npm run typecheck` under clients/web), not the repo
        # root where that command would not resolve. Defaults to project.path.
        run_dir = cwd or project.path
        logger.info(f"Running: {command} (cwd={run_dir}, timeout={timeout}s)")
        activation = (
            project.env_manager.activate_command() if project.env_manager else None
        )
        # Governance gate + audit trail (both no-ops when off: governance_policy
        # is None unless orchestrator.governance is set; audit only appends).
        # getattr-guarded so a lightweight project stub without these subsystems
        # still executes commands unchanged.
        policy = getattr(project, "governance_policy", None)
        audit = getattr(project, "audit_trail", None)
        success, output = _run_cmd(
            command, run_dir, activation, timeout, policy=policy, audit=audit
        )
        # A timeout is an environment signal (machine under load), not a code
        # failure. A slow-but-correct build wrongly marked "failing" poisons the
        # baseline and every gate after it (observed: an untouched, dependency-
        # free stub timing out at 120s under a competing compile, failing the
        # whole task). Retry once at an extended timeout so a transient load
        # spike self-heals; a genuine hang times out again and legitimately fails.
        if not success and output and output.startswith("Command timed out after"):
            extended = timeout * 2
            logger.warning(
                f"Command timed out after {timeout}s; retrying once at "
                f"{extended}s (transient load, not a code failure): {command}"
            )
            success, output = _run_cmd(
                command, run_dir, activation, extended, policy=policy, audit=audit
            )
        return success, output

    def _snapshot_files(
        self, project: Project, files: List[str]
    ) -> Dict[str, Optional[str]]:
        snapshot = {}
        for file_path in files:
            full_path = project.path / file_path
            if full_path.exists():
                try:
                    snapshot[file_path] = full_path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    snapshot[file_path] = None
            else:
                snapshot[file_path] = None
        return snapshot

    def _revert_files(
        self, project: Project, snapshot: Dict[str, Optional[str]]
    ) -> None:
        for file_path, content in snapshot.items():
            full_path = project.path / file_path
            if content is None:
                if full_path.exists():
                    full_path.unlink()
            else:
                write_file(full_path, content)

    def _record_success(self, task: Task, files: List[str]) -> None:
        self.scratchpad.record(
            category="pattern",
            discovery=f"Task {task.id} completed successfully",
            task_id=task.id,
            files=files,
            tags=[task.category],
        )

    def _get_processor_config(self, project: Project) -> dict:
        processors = project.config.get("task_processors", [])
        for p in processors:
            if p.get("type") == "markdown_planner":
                return p.get("settings", {})
        return {}
