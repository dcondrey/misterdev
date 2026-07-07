"""Run misterdev on a polyglot exercise and grade the result.

Sets up the exercise as a git repo (solution stub + test file + a project.yaml
scoping the solution file), drives ``misterdev build`` with the exercise
instructions as the goal, and grades by running the test. Repo setup is injected
so the harness is unit-testable offline.
"""

import json
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from .grader import GradeResult, grade
from .instance import PolyglotInstance

# Build files a compiled language needs, carried over when present.
_BUILD_META = (
    "Cargo.toml",
    "go.mod",
    "package.json",
    "build.gradle",
    "CMakeLists.txt",
)

# Flags misterdev's build parser (misterdev.core.modes.parse_flags) recognizes.
# Stripped from the free-text goal so a stray "--budget"/"--focus" token in an
# exercise prompt cannot be swallowed as a build flag (and eat the next token).
_BUILD_FLAG_TOKENS = frozenset(
    {
        "--budget",
        "--commit",
        "--no-verify",
        "--no-suggest",
        "--dry-run",
        "--interactive",
        "-i",
        "--parallel",
        "--no-rollback",
        "--allow-dirty",
        "--focus",
        "--max-tasks",
    }
)


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
    """Run a fixed git command in ``repo``, bounded and non-silent.

    Raises on a non-zero exit or a hang (60s cap) so run_instance records the
    exercise as a reasoned failure instead of the suite blocking on a stuck git
    or silently proceeding from a half-initialized repo.
    """
    proc = subprocess.run(
        f"git {args}",
        shell=True,
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"git {args} failed ({proc.returncode}): {detail}")


def _stage_base_files(repo: Path, instance: PolyglotInstance) -> None:
    """Stage exactly the exercise's own files (explicit paths, never ``git add -A``)."""
    rels = [
        r
        for r in [
            "project.yaml",
            *instance.solution_files,
            *instance.test_files,
            *_BUILD_META,
        ]
        if (repo / r).exists()
    ]
    if rels:
        _git(repo, "add -- " + " ".join(shlex.quote(r) for r in rels))


def _sanitize_goal(instructions: str) -> str:
    """Drop build-flag lookalike tokens from a free-text goal before it is merged
    with the real build args (which share one whitespace-split arg string).

    Only whole flag-lookalike tokens are removed; all other text (newlines, code
    indentation, paragraph structure) is preserved verbatim. When a flag is
    removed mid-line its preceding space is dropped too so no double gap remains,
    but a preceding newline is kept so line structure is never collapsed.
    """
    pieces: List[str] = []
    i, n = 0, len(instructions)
    while i < n:
        j = i
        if instructions[i].isspace():
            while j < n and instructions[j].isspace():
                j += 1
            pieces.append(instructions[i:j])
        else:
            while j < n and not instructions[j].isspace():
                j += 1
            token = instructions[i:j]
            if token in _BUILD_FLAG_TOKENS:
                if pieces and pieces[-1].isspace() and "\n" not in pieces[-1]:
                    pieces.pop()
            else:
                pieces.append(token)
        i = j
    return "".join(pieces)


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
        # Carry over build files a compiled language needs when present.
        for meta in _BUILD_META:
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
    ~$0 benchmark run — and disables the reflection loop, whose extra per-failure
    model call would otherwise spend the run's budget on something other than the
    edits. Scalar values are JSON-encoded (a valid YAML double-quoted scalar) so
    a name/model containing a quote or backslash never emits invalid YAML.
    """
    cfg = (
        f"name: {json.dumps(instance.name)}\n"
        f"language: {json.dumps(instance.language)}\n"
        f"test_command: {json.dumps(instance.test_command)}\n"
    )
    if model:
        cfg += (
            "llm:\n"
            f"  model: {json.dumps(model)}\n"
            '  provider: "openrouter"\n'
            "  failover: []\n"
            "  dynamic_selection: false\n"
            "orchestrator:\n"
            "  reflection: false\n"
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
        _stage_base_files(repo, instance)
        _git(repo, 'commit -q -m "polyglot base" --allow-empty')

        if orchestrator is None:
            from misterdev.agent import ProjectOrchestrator

            orchestrator = ProjectOrchestrator()
        goal = _sanitize_goal(instance.instructions)
        orchestrator.build(str(repo), f"{goal} {build_args}")

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
