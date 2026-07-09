"""Discover, run, and grade native-language (Swift, C#) exercises.

Swift and C# were the two toolchains the polyglot harness never supported, so
they had no end-to-end validation of misterdev at all. This harness closes that
gap the same way the polyglot one does: set up an exercise as a work dir (stub
solution + graded test + package manifest), drive misterdev to edit the stub,
then grade by running the exercise's own test command.

Design mirrors :mod:`evaluation.polyglot`: the misterdev-driving step is an
INJECTABLE ``run_one`` callable (default: a real ``ProjectOrchestrator`` drive)
and the pass/fail decision is a small pure function (:func:`grade_output`) so
the discovery + grading logic is unit-testable offline with no swift/dotnet
toolchain and no model cost.
"""

import math
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from misterdev.utils.process import kill_process_group

# Per-language test command, run from the exercise directory. Swift Package
# Manager and the .NET CLI both grade with a single sub-command whose exit code
# is authoritative; the printed summary is used only to corroborate/explain.
TEST_COMMANDS: Dict[str, str] = {
    "swift": "swift test",
    "csharp": "dotnet test",
}


@dataclass
class NativeExercise:
    """One native-language exercise.

    ``solution_files`` are the stubs misterdev edits; ``test_files`` are the
    graded tests it must make pass (and must not weaken). ``instructions`` is the
    prompt shown as the build goal; the test files are never described to the
    model.
    """

    name: str
    language: str
    solution_files: List[str]
    test_files: List[str]
    instructions: str = ""
    test_command: str = ""

    def __post_init__(self):
        if not self.test_command:
            self.test_command = TEST_COMMANDS.get(self.language, "")
            if not self.test_command:
                raise ValueError(
                    f"no test command known for language {self.language!r}; "
                    f"supported: {sorted(TEST_COMMANDS)}"
                )


@dataclass
class GradeResult:
    """Outcome of grading one exercise."""

    resolved: bool
    output: str = ""
    error: str = ""


@dataclass
class RunResult:
    """Outcome of running one exercise end to end."""

    name: str
    language: str
    resolved: bool
    duration_s: float = 0.0
    error: str = ""
    cost: float = 0.0  # dollars misterdev spent (best-effort)
    grade: Optional[GradeResult] = None


def grade_output(returncode: int, output: str) -> GradeResult:
    """Decide pass/fail from a finished test run's exit code and output.

    Exit code is the ground truth (both ``swift test`` and ``dotnet test`` exit
    non-zero on any test failure or build error). The printed summary is only
    used to catch the one case the exit code cannot: a runner that exits 0 while
    still reporting failures — ``dotnet test`` in particular has shipped configs
    where a non-fatal path leaves the code at 0 despite ``Failed!`` in the
    banner. So a run is RESOLVED iff it exits 0 AND the output shows no failure
    marker. Kept pure (no subprocess) so it is unit-tested against canned
    ``swift test`` / ``dotnet test`` strings with no toolchain present.
    """
    text = output or ""
    lowered = text.lower()
    # xUnit / dotnet:  "Failed! - Failed: 1, ..."
    # SwiftPM/XCTest:  "Test Suite '...' failed"  /  "build failed"
    has_failure = "failed!" in lowered or "build failed" in lowered
    if "test suite" in lowered and "' failed" in lowered:
        has_failure = True
    if returncode != 0:
        return GradeResult(False, output=text, error=f"tests exited {returncode}")
    if has_failure:
        return GradeResult(
            False, output=text, error="tests exited 0 but reported failures"
        )
    return GradeResult(True, output=text)


def grade(
    exercise_dir: str,
    exercise: NativeExercise,
    env_activate: Optional[str] = None,
    timeout: int = 900,
) -> GradeResult:
    """Run the exercise's test command in ``exercise_dir`` and report resolution.

    A missing test file, a timeout, or an unlaunchable command is a non-resolved
    result with a reason rather than a crash, so a suite run never aborts on one
    bad exercise. Native compiles are slow (SwiftPM/MSBuild cold builds), hence a
    higher default timeout than the polyglot grader.
    """
    root = Path(exercise_dir)
    if exercise.test_files and not any(
        (root / f).exists() for f in exercise.test_files
    ):
        return GradeResult(False, error="no test file present to grade against")
    cmd = exercise.test_command
    full = f"{env_activate} && {cmd}" if env_activate else cmd
    # start_new_session isolates the command in its own process group so a
    # timeout SIGKILLs the whole tree (swiftpm/msbuild workers) instead of
    # leaving grandchildren orphaned holding the build lock.
    try:
        proc = subprocess.Popen(
            full,
            shell=True,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        )
    except OSError as e:
        return GradeResult(False, error=f"could not run test command: {e}")
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_process_group(proc)
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        return GradeResult(False, error=f"test command timed out after {timeout}s")
    # Bound each stream before concatenating so a runaway build cannot materialize
    # a multi-MB string just to keep the tail that carries the failure summary.
    output = ((out or "")[-4000:] + (err or "")[-4000:])[-4000:]
    return grade_output(proc.returncode, output)


@dataclass
class SuiteReport:
    """Aggregate outcome of a suite run — the pass@1 number the harness produces."""

    results: List[RunResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def resolved(self) -> int:
        return sum(1 for r in self.results if r.resolved)

    @property
    def resolved_rate(self) -> float:
        return self.resolved / self.total if self.total else 0.0

    @property
    def cost(self) -> float:
        """Total dollars misterdev spent across the suite (best-effort sum)."""
        return math.fsum(getattr(r, "cost", 0.0) or 0.0 for r in self.results)

    def by_language(self) -> dict:
        out: dict = {}
        for r in self.results:
            agg = out.setdefault(r.language, [0, 0])
            agg[1] += 1
            if r.resolved:
                agg[0] += 1
        return out

    def to_dict(self) -> dict:
        """Machine-readable form: aggregate rates plus every per-instance result."""
        return {
            "total": self.total,
            "resolved": self.resolved,
            "resolved_rate": self.resolved_rate,
            "cost": self.cost,
            "results": [
                {
                    "name": r.name,
                    "language": r.language,
                    "resolved": r.resolved,
                    "duration_s": r.duration_s,
                    "error": r.error,
                    "cost": r.cost,
                }
                for r in self.results
            ],
        }

    def summary(self) -> str:
        lines = [
            f"Native: {self.resolved}/{self.total} resolved ({self.resolved_rate:.1%})"
        ]
        for lang, (ok, tot) in sorted(self.by_language().items()):
            lines.append(f"  {lang}: {ok}/{tot}")
        for r in self.results:
            mark = "PASS" if r.resolved else "FAIL"
            tail = f" — {r.error}" if r.error else ""
            lines.append(
                f"  [{mark}] {r.language}/{r.name} ({r.duration_s:.0f}s){tail}"
            )
        return "\n".join(lines)


def discover_exercises(
    root: str,
    languages: Optional[List[str]] = None,
    limit: Optional[int] = None,
    only: Optional[List[str]] = None,
) -> List[NativeExercise]:
    """Find exercises under ``root/<language>/<slug>/`` and load each.

    Each language dir holds one sub-dir per exercise slug; an exercise is loaded
    by reading solution/test file lists from its manifest and prompt from
    ``instructions.md`` (see :func:`load_local_exercise`). ``languages`` filters
    which language trees to include (default: the supported set); ``only``
    restricts to specific slugs for targeted re-runs; ``limit`` caps the total.

    limit=0 means zero exercises (not "all"); a negative limit is clamped to
    empty rather than silently dropping the last exercise via a ``[:-1]`` slice.
    """
    base = Path(root)
    langs = languages or sorted(TEST_COMMANDS)
    slugs = set(only) if only else None
    found: List[NativeExercise] = []
    for lang in langs:
        lang_dir = base / lang
        if not lang_dir.is_dir():
            continue
        for ex in sorted(p for p in lang_dir.iterdir() if p.is_dir()):
            if slugs is not None and ex.name not in slugs:
                continue
            found.append(load_local_exercise(str(ex), lang))
    if limit is not None:
        found = found[: max(limit, 0)]
    return found


def load_local_exercise(exercise_dir: str, language: str) -> NativeExercise:
    """Build an exercise from a fixture directory.

    Reads ``exercise.json`` for the solution/test file lists (falling back to a
    conventional single stub/test pair when absent) and ``instructions.md`` for
    the prompt. Raises if neither a manifest nor a discoverable layout exists, so
    a malformed fixture fails loudly at load rather than scoring as empty.
    """
    import json

    root = Path(exercise_dir)
    manifest = root / "exercise.json"
    if manifest.exists():
        config = json.loads(manifest.read_text(encoding="utf-8"))
        files = config.get("files", {})
        solution = list(files.get("solution", []))
        test = list(files.get("test", []))
        instructions = config.get("instructions", "")
    else:
        solution, test, instructions = [], [], ""
    docs = root / "instructions.md"
    if docs.exists():
        instructions = docs.read_text(encoding="utf-8")
    if not solution or not test:
        raise ValueError(
            f"exercise {root.name!r} has no solution/test files declared in "
            f"exercise.json"
        )
    return NativeExercise(
        name=root.name,
        language=language,
        solution_files=solution,
        test_files=test,
        instructions=instructions,
    )


# Setup that copies a fixture exercise dir into a fresh work dir; injected so the
# suite is testable with a fake that writes files directly.
PrepareRepo = Callable[[NativeExercise, Path], Path]
# The misterdev-driving step: given a prepared work dir, solve + grade it.
RunOne = Callable[[NativeExercise, Path], RunResult]


def prepare_from_source(source_dir: str) -> PrepareRepo:
    """Repo-setup that copies a fixture exercise dir into the work dir.

    Copies every file under ``source_dir`` (solution stub, test, and the package
    manifest a compiled native build needs — Package.swift / *.csproj) so the
    work dir is buildable, then returns the destination. Files not declared in
    the manifest are carried too because SwiftPM/MSBuild need the whole package
    layout, not just the graded pair. Toolchain build output (``.build`` for
    SwiftPM, ``bin``/``obj`` for MSBuild) is skipped: copying it wastes I/O and
    can poison the fresh build with another machine's stale, path-baked cache.
    """
    import shutil

    src = Path(source_dir)
    _skip_dirs = {".build", "bin", "obj", ".git"}

    def _prepare(exercise: NativeExercise, dest: Path) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        for s in src.rglob("*"):
            rel = s.relative_to(src)
            if any(part in _skip_dirs for part in rel.parts):
                continue
            if s.is_dir():
                continue
            d = dest / rel
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
        return dest

    return _prepare


def default_run_one(
    prepare_repo: PrepareRepo,
    workdir: str,
    *,
    orchestrator=None,
    build_args: str = "--budget 2 --allow-dirty --no-suggest",
    env_activate: Optional[str] = None,
    grade_timeout: int = 900,
) -> RunOne:
    """Build the default ``run_one``: a real misterdev drive + test-command grade.

    ``orchestrator`` defaults to a real ``ProjectOrchestrator`` imported lazily
    so importing this harness never pulls the whole engine (and offline tests can
    inject a fake instead). Any setup/build failure is captured as a non-resolved
    result with a reason, so a suite run never aborts on one bad exercise.
    """

    def _run(exercise: NativeExercise, dest: Path) -> RunResult:
        nonlocal orchestrator
        start = time.time()
        try:
            prepare_repo(exercise, dest)
            if orchestrator is None:
                from misterdev.agent import ProjectOrchestrator

                orchestrator = ProjectOrchestrator()
            orchestrator.build(str(dest), f"{exercise.instructions} {build_args}")
            result = grade(
                str(dest),
                exercise,
                env_activate=env_activate,
                timeout=grade_timeout,
            )
            return RunResult(
                name=exercise.name,
                language=exercise.language,
                resolved=result.resolved,
                duration_s=time.time() - start,
                error=result.error,
                cost=float(getattr(orchestrator, "last_build_cost", 0.0) or 0.0),
                grade=result,
            )
        except Exception as e:  # one bad exercise must not sink the suite
            return RunResult(
                name=exercise.name,
                language=exercise.language,
                resolved=False,
                duration_s=time.time() - start,
                error=f"{type(e).__name__}: {e}",
            )

    return _run


def run_suite(
    root: str,
    workdir: str,
    *,
    languages: Optional[List[str]] = None,
    limit: Optional[int] = None,
    only: Optional[List[str]] = None,
    run_one: Optional[RunOne] = None,
    progress=None,
    **run_kwargs,
) -> SuiteReport:
    """Discover exercises under ``root`` and solve + grade each through misterdev.

    Sequential by design (each native build is itself resource-heavy). ``run_one``
    is injectable — the default is a real misterdev drive (:func:`default_run_one`)
    built over :func:`prepare_from_source`; tests pass a fake to exercise the
    discovery + aggregation path offline. ``progress`` is called with each
    :class:`RunResult` as it completes.
    """
    report = SuiteReport()
    for exercise in discover_exercises(root, languages, limit, only):
        source_dir = str(Path(root) / exercise.language / exercise.name)
        dest = Path(workdir) / f"{exercise.language}-{exercise.name}"
        runner = run_one or default_run_one(
            prepare_from_source(source_dir), workdir, **run_kwargs
        )
        result = runner(exercise, dest)
        report.results.append(result)
        if progress is not None:
            progress(result)
    return report
