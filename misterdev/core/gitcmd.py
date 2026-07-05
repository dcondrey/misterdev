"""Shared runner for one-off, read-only git commands.

Several modules run a single git command (``rev-parse``, ``diff``, ``log``,
``ls-files``) the same way: ``subprocess.run`` with ``shell=True`` against a repo
directory, capturing text output, bounded by a short timeout, and degrading to a
no-op on any failure. This centralizes that boilerplate; callers keep their own
post-processing (which legitimately differs — a sha, a diff body, stat text).

Only for fixed, read-only commands. Commands that interpolate untrusted values
must NOT use this shell path; they belong in the list-form helpers that avoid the
shell entirely.
"""

import subprocess
from pathlib import Path
from typing import Optional


def run_git(
    command: str, cwd: Path, timeout: float = 30
) -> Optional[subprocess.CompletedProcess]:
    """Run a fixed git ``command`` (shell string) in ``cwd``.

    Returns the ``CompletedProcess`` (with ``.returncode`` and ``.stdout``), or
    ``None`` when it could not run at all — missing git, a timeout, or an OS
    error. Never raises, so callers can treat ``None`` (and a non-zero
    ``returncode``) as "no result" and degrade gracefully.
    """
    try:
        return subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
