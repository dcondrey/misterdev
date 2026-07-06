"""Run a suite of SWE-bench instances and report the resolved rate."""

from dataclasses import dataclass, field
from typing import List, Optional

from .instance import SWEBenchInstance
from .runner import RunResult, run_instance


@dataclass
class SuiteReport:
    """Aggregate outcome of a suite run — the number the harness exists to produce."""

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
    def errored(self) -> int:
        return sum(1 for r in self.results if r.error and not r.resolved)

    def summary(self) -> str:
        lines = [
            f"SWE-bench: {self.resolved}/{self.total} resolved "
            f"({self.resolved_rate:.1%})",
        ]
        for r in self.results:
            mark = "PASS" if r.resolved else "FAIL"
            tail = f" — {r.error}" if r.error else ""
            lines.append(f"  [{mark}] {r.instance_id} ({r.duration_s:.0f}s){tail}")
        return "\n".join(lines)


def run_suite(
    instances: List[SWEBenchInstance],
    workdir: str,
    *,
    limit: Optional[int] = None,
    progress=None,
    **run_kwargs,
) -> SuiteReport:
    """Run each instance sequentially and collect a report.

    Sequential by design: each build is itself parallel and resource-heavy, so
    running instances concurrently would contend for CPU and muddy the timing
    signal. ``limit`` caps how many run; ``progress`` is an optional callback
    invoked with each :class:`RunResult` as it completes.
    """
    report = SuiteReport()
    for inst in instances[: limit or len(instances)]:
        result = run_instance(inst, workdir, **run_kwargs)
        report.results.append(result)
        if progress is not None:
            progress(result)
    return report
