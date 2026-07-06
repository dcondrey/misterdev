"""Grade a polyglot exercise by running its own test command.

An exercise is RESOLVED when its test command exits 0 after misterdev has edited
the solution stub — the language-agnostic ground truth. Kept small and
unit-tested; the runner drives setup and misterdev, this decides pass/fail.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .instance import PolyglotInstance


@dataclass
class GradeResult:
    """Outcome of grading one exercise."""

    resolved: bool
    output: str = ""
    error: str = ""


def grade(
    exercise_dir: str,
    instance: PolyglotInstance,
    env_activate: Optional[str] = None,
    timeout: int = 600,
) -> GradeResult:
    """Run the exercise's test command in ``exercise_dir`` and report resolution.

    Resolved iff the test command exits 0. A timeout or a missing test file is a
    non-resolved result with a reason rather than a crash, so a suite run never
    aborts on one bad exercise.
    """
    root = Path(exercise_dir)
    if instance.test_files and not any(
        (root / f).exists() for f in instance.test_files
    ):
        return GradeResult(False, error="no test file present to grade against")
    cmd = instance.test_command
    full = f"{env_activate} && {cmd}" if env_activate else cmd
    try:
        proc = subprocess.run(
            full,
            shell=True,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return GradeResult(False, error=f"test command timed out after {timeout}s")
    output = (proc.stdout or "") + (proc.stderr or "")
    return GradeResult(resolved=proc.returncode == 0, output=output[-4000:])
