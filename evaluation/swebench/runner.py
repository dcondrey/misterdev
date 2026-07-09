"""Run misterdev on a SWE-bench instance and grade the result.

Sets up the task repo at its base commit, drives misterdev's build with the
issue as the goal, extracts the produced patch, and grades it. Repo setup is
injected (``prepare_repo``) so the same runner works against a real GitHub clone
or a local mirror, and so the harness logic is unit-testable offline.
"""

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .grader import GradeResult, grade
from .instance import SWEBenchInstance


@dataclass
class RunResult:
    """Outcome of running one instance end to end."""

    instance_id: str
    resolved: bool
    patch: str = ""
    duration_s: float = 0.0
    error: str = ""
    grade: Optional[GradeResult] = None


def _git(repo: Path, args: str) -> str:
    proc = subprocess.run(
        f"git {args}", shell=True, cwd=str(repo), capture_output=True, text=True
    )
    return (proc.stdout or "").strip()


def prepare_repo_clone(instance: SWEBenchInstance, dest: Path) -> Path:
    """Clone the instance repo into ``dest`` and check out its base commit.

    Uses a shallow fetch of the exact commit so a large history isn't pulled.
    Requires network + git; injected alternatives (a local mirror) are used in
    tests and offline runs.
    """
    dest.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{instance.repo}.git"
    subprocess.run(f"git init -q {dest}", shell=True, check=True)
    subprocess.run(f"git -C {dest} remote add origin {url}", shell=True, check=True)
    subprocess.run(
        f"git -C {dest} fetch -q --depth 1 origin {instance.base_commit}",
        shell=True,
        check=True,
    )
    subprocess.run(f"git -C {dest} checkout -q FETCH_HEAD", shell=True, check=True)
    return dest


def _write_project_yaml(repo: Path, instance: SWEBenchInstance) -> None:
    """Write a minimal project.yaml so misterdev can gate the build.

    The test_command here is the repo's OWN suite (never the hidden FAIL_TO_PASS
    tests, which arrive only via test_patch at grade time), so the model cannot
    see or game the tests it is judged on.
    """
    if (repo / "project.yaml").exists():
        return
    # spec_as_tests engages reproduce-then-fix: the model never sees the hidden
    # FAIL_TO_PASS tests, so a validated reproduction synthesized from the issue
    # (kept only if it fails on the clean tree) is the one gate that directly
    # targets the judged behavior. Advisory (non-blocking) so a stray repro can't
    # burn the per-instance budget, but it still directs the edit as the concrete
    # objective. The repo's own suite stays the authoritative regression gate.
    cfg = (
        f'name: "{instance.instance_id}"\n'
        f'language: "{instance.language}"\n'
        f'test_command: "{instance.test_command}"\n'
        "orchestrator:\n"
        "  spec_as_tests: true\n"
    )
    (repo / "project.yaml").write_text(cfg, encoding="utf-8")


PrepareRepo = Callable[[SWEBenchInstance, Path], Path]


def run_instance(
    instance: SWEBenchInstance,
    workdir: str,
    *,
    orchestrator=None,
    build_args: str = "--budget 5 --allow-dirty --no-suggest",
    env_activate: Optional[str] = None,
    prepare_repo: PrepareRepo = prepare_repo_clone,
    grade_timeout: int = 1800,
) -> RunResult:
    """Set up ``instance``, run misterdev, and grade the patch.

    ``orchestrator`` defaults to a real ``ProjectOrchestrator`` (imported lazily
    so importing the harness never pulls the whole engine); a stub is injected in
    tests. Any setup/build failure is captured as a non-resolved result with a
    reason rather than raised, so a suite run never aborts on one bad instance.
    """
    start = time.time()
    repo = Path(workdir) / instance.instance_id
    try:
        prepare_repo(instance, repo)
        _write_project_yaml(repo, instance)
        for cmd in instance.setup_commands:
            subprocess.run(cmd, shell=True, cwd=str(repo))
        # Commit a clean base so the model's patch is exactly diff(base..HEAD).
        _git(repo, "add -A")
        _git(repo, 'commit -q -m "swebench base" --allow-empty')
        base = _git(repo, "rev-parse HEAD")

        if orchestrator is None:
            from misterdev.agent import ProjectOrchestrator

            orchestrator = ProjectOrchestrator()
        orchestrator.build(str(repo), f"{instance.problem_statement} {build_args}")

        patch = _git(repo, f"diff {base} HEAD")
        result = grade(
            str(repo), instance, env_activate=env_activate, timeout=grade_timeout
        )
        return RunResult(
            instance_id=instance.instance_id,
            resolved=result.resolved,
            patch=patch,
            duration_s=time.time() - start,
            error=result.error,
            grade=result,
        )
    except Exception as e:  # one bad instance must not sink the suite
        return RunResult(
            instance_id=instance.instance_id,
            resolved=False,
            duration_s=time.time() - start,
            error=f"{type(e).__name__}: {e}",
        )
