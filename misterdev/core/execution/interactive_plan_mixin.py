"""InteractivePlanMixin — user-facing interactive planning helpers.

Extracted from agent.py. Covers goal selection, plan confirmation, dirty-tree
detection, and the work-type→mode mapping shared by the interactive and
propose/execute flows.
"""

import subprocess
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.prompt import Prompt

from misterdev.core.modes import BuildMode, resolve_mode
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)
_console = Console()


class InteractivePlanMixin:
    _WORK_TYPE_MODES = {
        "debug": BuildMode.DEBUG,
        "complete": BuildMode.COMPLETE,
        "feature": BuildMode.SMART,
        "refactor": BuildMode.SMART,
        "test": BuildMode.SMART,
        "docs": BuildMode.SMART,
    }

    def _choose_goal(self, recs: list) -> tuple[Optional[str], BuildMode]:
        """Present recommendations and return the chosen (goal, mode).

        Returns (None, _) if the user quits. A free-text goal resolves its own
        mode; a picked recommendation carries the advisor's work_type.
        """
        if recs:
            _console.print("\n[bold]Recommended work:[/]")
            for i, r in enumerate(recs, 1):
                _console.print(
                    f"  [cyan]{i}[/]. {r.title} [dim]({r.work_type}) — {r.rationale}[/]"
                )
        _console.print("\nEnter a number to pick, type your own goal, or 'q' to quit.")
        choice = Prompt.ask("Goal").strip()
        if not choice or choice.lower() in ("q", "quit"):
            return None, BuildMode.SMART
        if choice.isdigit() and recs:
            idx = int(choice) - 1
            if 0 <= idx < len(recs):
                r = recs[idx]
                return r.title, self._WORK_TYPE_MODES.get(r.work_type, BuildMode.SMART)
            _console.print("[yellow]Out of range; treating input as a goal.[/]")
        return choice, resolve_mode(choice, Path("."))

    def _confirm(self, question: str) -> bool:
        """Ask a yes/no question; defaults to no."""
        return Prompt.ask(f"{question} [y/N]", default="n").strip().lower() in (
            "y",
            "yes",
        )

    def _working_tree_dirty(self, project) -> str:
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
