"""Run misterdev on a polyglot exercise and grade the result.

Sets up the exercise as a git repo (solution stub + test file + a project.yaml
scoping the solution file), drives ``misterdev build`` with the exercise
instructions as the goal, and grades by running the test. Repo setup is injected
so the harness is unit-testable offline.
"""

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .grader import GradeResult, grade
from .instance import PolyglotInstance


@dataclass
class RunResult:
    """Outcome of running one exercise end to end."""

    name: str
    language: str
    resolved: bool
    duration_s: float = 0.0
    error: str = ""
    grade: Optional[GradeResult] = None


def _git(repo: Path, args: str) -> None:
    subprocess.run(f"git {args}", shell=True, cwd=str(repo), capture_output=True)


def prepare_from_source(source_dir: str) -> Callable[[PolyglotInstance, Path], Path]:
    """Repo-setup that copies a checked-out exercise dir into the work dir.

    ``source_dir`` is the exercise directory in a local polyglot-benchmark
    checkout; the returned callable copies solution + test files (and the
    solution's build metadata for compiled languages) into a fresh git repo.
    """

    src = Path(source_dir)

    def _prepare(instance: PolyglotInstance, dest: Path) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        for rel in instance.solution_files + instance.test_files:
            s = src / rel
            d = dest / rel
            if s.exists():
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(s, d)
        # Carry over build files a compiled language needs (Cargo.toml, go.mod,
        # package.json, build.gradle, CMakeLists.txt) when present.
        for meta in (
            "Cargo.toml",
            "go.mod",
            "package.json",
            "build.gradle",
            "CMakeLists.txt",
        ):
            m = src / meta
            if m.exists():
                shutil.copy2(m, dest / meta)
        return dest

    return _prepare


def _write_project_yaml(
    repo: Path, instance: PolyglotInstance, model: Optional[str] = None
) -> None:
    """Write a project.yaml so misterdev's test gate runs the exercise's tests.

    ``allow_test_edits`` stays off (the default), so misterdev's test-tamper gate
    protects the graded test file — the model must fix the solution, not the test.
    When ``model`` is given it pins that single model with failover and dynamic
    selection off, so a free-model run cannot escalate to a paid model — a real
    ~$0 benchmark run.
    """
    cfg = (
        f'name: "{instance.name}"\n'
        f'language: "{instance.language}"\n'
        f'test_command: "{instance.test_command}"\n'
    )
    if model:
        cfg += (
            "llm:\n"
            f'  model: "{model}"\n'
            '  provider: "openrouter"\n'
            "  failover: []\n"
            "  dynamic_selection: false\n"
        )
    (repo / "project.yaml").write_text(cfg, encoding="utf-8")


PrepareRepo = Callable[[PolyglotInstance, Path], Path]


def run_instance(
    instance: PolyglotInstance,
    workdir: str,
    prepare_repo: PrepareRepo,
    *,
    orchestrator=None,
    model: Optional[str] = None,
    build_args: str = "--budget 2 --allow-dirty --no-suggest",
    env_activate: Optional[str] = None,
    grade_timeout: int = 600,
) -> RunResult:
    """Set up ``instance``, run misterdev, and grade the solution.

    ``orchestrator`` defaults to a real ``ProjectOrchestrator`` (imported lazily
    so importing the harness never pulls the whole engine). Any setup/build
    failure is captured as a non-resolved result with a reason, so a suite run
    never aborts on one bad exercise.
    """
    start = time.time()
    repo = Path(workdir) / f"{instance.language}-{instance.name}"
    try:
        prepare_repo(instance, repo)
        _write_project_yaml(repo, instance, model=model)
        _git(repo, "init -q")
        _git(repo, "add -A")
        _git(repo, 'commit -q -m "polyglot base" --allow-empty')

        if orchestrator is None:
            from misterdev.agent import ProjectOrchestrator

            orchestrator = ProjectOrchestrator()
        orchestrator.build(str(repo), f"{instance.instructions} {build_args}")

        result = grade(
            str(repo), instance, env_activate=env_activate, timeout=grade_timeout
        )
        return RunResult(
            name=instance.name,
            language=instance.language,
            resolved=result.resolved,
            duration_s=time.time() - start,
            error=result.error,
            grade=result,
        )
    except Exception as e:  # one bad exercise must not sink the suite
        return RunResult(
            name=instance.name,
            language=instance.language,
            resolved=False,
            duration_s=time.time() - start,
            error=f"{type(e).__name__}: {e}",
        )
