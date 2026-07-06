"""Run a suite of polyglot exercises and report the resolved rate."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .instance import load_local_exercise
from .runner import RunResult, prepare_from_source, run_instance


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

    def by_language(self) -> dict:
        out: dict = {}
        for r in self.results:
            agg = out.setdefault(r.language, [0, 0])
            agg[1] += 1
            if r.resolved:
                agg[0] += 1
        return out

    def summary(self) -> str:
        lines = [
            f"Polyglot: {self.resolved}/{self.total} resolved "
            f"({self.resolved_rate:.1%})"
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
    benchmark_dir: str,
    languages: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> List[tuple]:
    """Find (exercise_dir, language) pairs in a polyglot-benchmark checkout.

    ``languages`` filters which language trees to include (default: all found);
    ``limit`` caps the total. Returns pairs ready for :func:`load_local_exercise`.
    """
    root = Path(benchmark_dir)
    langs = languages or ["cpp", "go", "java", "javascript", "python", "rust"]
    found: List[tuple] = []
    for lang in langs:
        practice = root / lang / "exercises" / "practice"
        if not practice.is_dir():
            continue
        for ex in sorted(p for p in practice.iterdir() if p.is_dir()):
            found.append((str(ex), lang))
    return found[: limit or len(found)]


def run_suite(
    benchmark_dir: str,
    workdir: str,
    *,
    languages: Optional[List[str]] = None,
    limit: Optional[int] = None,
    progress=None,
    **run_kwargs,
) -> SuiteReport:
    """Discover exercises under ``benchmark_dir`` and run each through misterdev.

    Sequential by design (each build is itself resource-heavy). ``progress`` is
    called with each :class:`RunResult` as it completes.
    """
    report = SuiteReport()
    for exercise_dir, lang in discover_exercises(benchmark_dir, languages, limit):
        instance = load_local_exercise(exercise_dir, lang)
        result = run_instance(
            instance, workdir, prepare_from_source(exercise_dir), **run_kwargs
        )
        report.results.append(result)
        if progress is not None:
            progress(result)
    return report
