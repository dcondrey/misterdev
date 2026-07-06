"""Grade a candidate fix against a SWE-bench task's own tests.

A task is RESOLVED when, after the model's patch and then the task's test_patch
are applied, every FAIL_TO_PASS test passes and every PASS_TO_PASS test still
passes. This is the objective definition of success — the ground truth the whole
harness reports against — so it is kept small, dependency-free, and unit-tested.
"""

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .instance import SWEBenchInstance

# pytest's `-rA` short summary lists one line per test: "PASSED path::name".
_SUMMARY_LINE = re.compile(r"^(PASSED|FAILED|ERROR|XFAIL|XPASS)\s+(\S+)")
# A test counts as "passing" only for these outcomes.
_PASSING = {"PASSED", "XFAIL"}


@dataclass
class GradeResult:
    """Outcome of grading one instance."""

    resolved: bool
    fail_to_pass: Dict[str, bool] = field(default_factory=dict)
    pass_to_pass: Dict[str, bool] = field(default_factory=dict)
    error: str = ""


def _run(cmd: str, cwd: Path, env_activate: Optional[str], timeout: int) -> str:
    """Run a shell command and return combined stdout+stderr (empty on timeout)."""
    full = f"{env_activate} && {cmd}" if env_activate else cmd
    try:
        proc = subprocess.run(
            full,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ""
    return (proc.stdout or "") + (proc.stderr or "")


def apply_patch(repo_dir: Path, patch_text: str) -> bool:
    """Apply a unified diff to ``repo_dir`` (git apply, with a 3-way fallback).

    An empty patch is a no-op success. Returns False when the patch does not
    apply cleanly, which the grader reports rather than silently mis-scoring.
    """
    if not patch_text.strip():
        return True
    for extra in ("", "--3way "):
        proc = subprocess.run(
            f"git apply --whitespace=nowarn {extra}-",
            shell=True,
            cwd=str(repo_dir),
            input=patch_text,
            text=True,
            capture_output=True,
        )
        if proc.returncode == 0:
            return True
    return False


def _run_tests(
    repo_dir: Path,
    test_command: str,
    node_ids: List[str],
    env_activate: Optional[str],
    timeout: int,
) -> Dict[str, bool]:
    """Run the given test node ids and return {node_id: passed}.

    Every requested id starts False, so a test that errors, is not collected, or
    never appears in the summary counts as failing (never a silent pass)."""
    results = {n: False for n in node_ids}
    if not node_ids:
        return results
    quoted = " ".join(f"'{n}'" for n in node_ids)
    output = _run(f"{test_command} {quoted}", repo_dir, env_activate, timeout)
    for line in output.splitlines():
        m = _SUMMARY_LINE.match(line.strip())
        if not m:
            continue
        outcome, nid = m.group(1), m.group(2)
        if nid in results:
            results[nid] = outcome in _PASSING
    return results


def grade(
    repo_dir: str,
    instance: SWEBenchInstance,
    env_activate: Optional[str] = None,
    timeout: int = 1800,
) -> GradeResult:
    """Grade the working tree at ``repo_dir`` (model's patch already applied).

    Applies the task's test_patch on top, runs FAIL_TO_PASS and PASS_TO_PASS, and
    resolves only when every FAIL_TO_PASS passes and every PASS_TO_PASS passes.
    A missing test list, or a test_patch that will not apply, is a non-resolved
    result with a reason rather than a crash.
    """
    root = Path(repo_dir)
    if not instance.fail_to_pass:
        return GradeResult(False, error="instance has no FAIL_TO_PASS tests")
    if not apply_patch(root, instance.test_patch):
        return GradeResult(False, error="test_patch did not apply")

    all_ids = list(instance.fail_to_pass) + list(instance.pass_to_pass)
    outcomes = _run_tests(root, instance.test_command, all_ids, env_activate, timeout)
    ftp = {n: outcomes.get(n, False) for n in instance.fail_to_pass}
    ptp = {n: outcomes.get(n, False) for n in instance.pass_to_pass}
    resolved = all(ftp.values()) and all(ptp.values())
    return GradeResult(resolved=resolved, fail_to_pass=ftp, pass_to_pass=ptp)
