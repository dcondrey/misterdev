"""Add a dependency through the project's package manager.

The model-callable surface over :func:`dependency_add_command`: it resolves the
right manager from the manifest (cargo/npm/pnpm/yarn/bun/uv/pip/dotnet), then
runs the add as an argument list (never a shell string) so a package name can't
inject a command. A deliberate add is expected to touch the lock file; a refactor
never routes through here.
"""

import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Tuple

from misterdev.analyzers.project_analyzer.detection import dependency_add_command
from misterdev.logging_setup import setup_logger
from misterdev.tools.base_tool import BaseTool

logger = setup_logger(__name__)

# A package spec is a name, optionally scoped (@scope/pkg), dotted (.NET), or
# @-pinned (name@1.2). No shell metacharacters, whitespace, or version operators
# (<>=) that would both inject and misparse under a shell — reject those and let
# the caller pin the version by editing the manifest.
_VALID_PACKAGE = re.compile(r"^[A-Za-z0-9._@/+-]+$")


class DependencyTool(BaseTool):
    def _resolve(self, project: Any, package: str) -> Tuple[str, str]:
        """Return ``(command, "")`` to run, or ``("", reason)`` when it can't.

        Validation happens here so the reason is returned without ever building
        a command from an unsafe name.
        """
        if not package or not _VALID_PACKAGE.match(package):
            return "", f"invalid package name: {package!r}"
        cmd = dependency_add_command(Path(project.path), package)
        if cmd is None:
            return "", (
                "no recognized package manager for this project "
                "(or its manifest is edited by hand, e.g. SwiftPM)"
            )
        return cmd, ""

    def execute(
        self, project: Any, package: str, timeout: int = 120
    ) -> Tuple[bool, str]:
        """Add ``package`` via the project's manager. Returns (success, output)."""
        cmd, reason = self._resolve(project, package)
        if not cmd:
            return False, reason
        logger.info(f"Adding dependency: '{cmd}' in {project.path}")
        try:
            proc = subprocess.run(
                shlex.split(cmd),
                cwd=str(project.path),
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, f"dependency add timed out after {timeout}s"
        except OSError as e:
            return False, f"could not run '{cmd}': {e}"
        output = ((proc.stdout or "") + (proc.stderr or "")).strip()
        return proc.returncode == 0, output
