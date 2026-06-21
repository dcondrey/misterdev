"""Build mode system ported from /build skill.

Determines operating mode from user input and project state.
"""

from enum import Enum
from pathlib import Path
from typing import Optional


class BuildMode(str, Enum):
    DEBUG = "debug"
    COMPLETE = "complete"
    REVIEW = "review"
    CREATE = "create"
    SPEC = "spec"
    AUTO = "auto"
    SMART = "smart"


class BuildFlags:
    """Parsed flags from user input."""

    def __init__(
        self,
        budget: float = 100.0,
        commit: bool = False,
        no_verify: bool = False,
        no_suggest: bool = False,
        dry_run: bool = False,
        focus: Optional[str] = None,
        interactive: bool = False,
        parallel: bool = False,
        no_rollback: bool = False,
        allow_dirty: bool = False,
    ):
        self.budget = budget
        self.commit = commit
        self.no_verify = no_verify
        self.no_suggest = no_suggest
        self.dry_run = dry_run
        self.focus = focus
        self.interactive = interactive
        self.parallel = parallel
        self.no_rollback = no_rollback
        self.allow_dirty = allow_dirty

    def __repr__(self) -> str:
        return (
            f"BuildFlags(budget={self.budget}, commit={self.commit}, "
            f"dry_run={self.dry_run}, focus={self.focus}, interactive={self.interactive})"
        )


def parse_flags(args: list[str]) -> tuple[list[str], BuildFlags]:
    """Extract flags from args list, return (remaining_args, flags)."""
    remaining = []
    flags = BuildFlags()
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--budget" and i + 1 < len(args):
            flags.budget = float(args[i + 1])
            i += 2
        elif arg == "--commit":
            flags.commit = True
            i += 1
        elif arg == "--no-verify":
            flags.no_verify = True
            i += 1
        elif arg == "--no-suggest":
            flags.no_suggest = True
            i += 1
        elif arg == "--dry-run":
            flags.dry_run = True
            i += 1
        elif arg == "--interactive" or arg == "-i":
            flags.interactive = True
            i += 1
        elif arg == "--parallel":
            flags.parallel = True
            i += 1
        elif arg == "--no-rollback":
            flags.no_rollback = True
            i += 1
        elif arg == "--allow-dirty":
            flags.allow_dirty = True
            i += 1
        elif arg == "--focus" and i + 1 < len(args):
            flags.focus = args[i + 1]
            i += 2
        else:
            remaining.append(arg)
            i += 1
    return remaining, flags


def resolve_mode(prompt: str, project_path: Path) -> BuildMode:
    """Determine build mode from prompt content and project state.

    Resolution order (from /build skill):
      - "debug"           -> DEBUG
      - "complete"        -> COMPLETE
      - "review"          -> REVIEW
      - "new <desc>"      -> CREATE
      - path to .md file  -> SPEC
      - empty prompt      -> AUTO (project exists -> COMPLETE, else CREATE)
      - anything else     -> SMART
    """
    stripped = prompt.strip().lower()

    if stripped == "debug":
        return BuildMode.DEBUG
    if stripped == "complete":
        return BuildMode.COMPLETE
    if stripped == "review":
        return BuildMode.REVIEW
    if stripped.startswith("new "):
        return BuildMode.CREATE

    # Check if prompt is a path to an existing .md spec file
    if prompt.strip().endswith(".md"):
        spec_path = project_path / prompt.strip()
        if spec_path.exists():
            return BuildMode.SPEC

    if not stripped:
        # AUTO: detect from project state
        has_code = _project_has_code(project_path)
        return BuildMode.COMPLETE if has_code else BuildMode.CREATE

    return BuildMode.SMART


def _project_has_code(project_path: Path) -> bool:
    """Check if a project directory contains source code files."""
    code_extensions = {
        ".py",
        ".js",
        ".ts",
        ".rs",
        ".go",
        ".java",
        ".c",
        ".cpp",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".scala",
        ".zig",
    }
    for item in project_path.rglob("*"):
        if item.suffix in code_extensions and not _is_venv_or_vendor(item):
            return True
    return False


def _is_venv_or_vendor(path: Path) -> bool:
    """Exclude virtual environments and vendor directories."""
    exclude_dirs = {
        "venv",
        ".venv",
        "env",
        "node_modules",
        "vendor",
        "__pycache__",
        ".git",
        "target",
        "build",
        "dist",
    }
    return any(part in exclude_dirs for part in path.parts)
