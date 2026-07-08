"""The sandbox evaluator — the loop's ``evaluate`` adapter, score-without-trust.

Turns a proposed :class:`Mutation` into a :class:`FitnessScore` (or ``None``)
while never trusting the candidate: the diff is applied in an ISOLATED git
worktree, the full existing gate suite must pass before a cent is spent on the
benchmark, and correctness is backstopped by counting regressions against a
baseline pass-set. Every step that touches source, runs gates, or spends money is
an INJECTED callable, so the orchestration is fully testable with fakes and each
dangerous step is individually swappable and guarded (tier-2 wires the real ones).

Order matters and encodes the cost discipline:
1. Re-check the guardrail (defense-in-depth; the loop checks too).
2. Apply the diff in a worktree; guarantee teardown on every path.
3. Run the gates — a FAIL returns ``None`` and skips the benchmark entirely, so a
   build-breaking candidate costs nothing.
4. Run the (proxy) benchmark; compute regressions vs baseline; return the score.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Set, Tuple

from misterdev.logging_setup import setup_logger

from .fitness import FitnessScore
from .guardrail import assert_mutation_allowed
from .loop import Mutation

logger = setup_logger(__name__)


def _instance_id(result) -> str:
    return str(
        getattr(result, "instance_id", None) or getattr(result, "name", None) or ""
    )


@dataclass
class SandboxEvaluator:
    """Injectable ``evaluate`` for :class:`EvolutionLoop`; safe by construction."""

    apply: Callable[[Mutation], Callable[[], None]]
    gates: Callable[[], bool]
    benchmark: Callable[[], Tuple[object, float]]
    baseline_passed: Set[str] = field(default_factory=set)

    def __call__(self, mutation: Mutation) -> Optional[FitnessScore]:
        # Defense-in-depth: refuse a walled-off/escaping target even if this
        # evaluator is driven outside the loop's guardrail.
        assert_mutation_allowed(mutation.paths)

        teardown = self.apply(mutation)
        try:
            if not self.gates():
                logger.info("Sandbox: gates failed; candidate discarded unscored.")
                return None
            report, cost = self.benchmark()
            # Expose the per-instance report so a held-out promotion gate (L4) can
            # partition this candidate's results into DERIVE/HOLDOUT pools; the
            # aggregate FitnessScore alone cannot be split.
            self.last_report = report
            regressions = self._count_regressions(report)
            return FitnessScore.from_report(report, cost=cost, regressions=regressions)
        finally:
            # The worktree is always torn down — on gate failure, on a benchmark
            # raise, and on success — so candidates never leak worktrees or disk.
            try:
                teardown()
            except Exception as e:  # cleanup must never mask the real outcome
                logger.warning(f"Sandbox teardown failed (non-fatal): {e}")

    def _count_regressions(self, report) -> int:
        """Instances that PASSED at baseline and FAIL now — the correctness
        backstop the fitness function hard-gates on."""
        passed_now = {
            _instance_id(r)
            for r in getattr(report, "results", [])
            if bool(getattr(r, "resolved", False))
        }
        return sum(1 for iid in self.baseline_passed if iid and iid not in passed_now)
