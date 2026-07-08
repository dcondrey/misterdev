"""The micro-evaluator — a cheap, dense screen before the expensive oracle.

The full benchmark is the correctness oracle, but it is far too expensive and
coarse to steer an evolutionary search: minutes and dollars per candidate, and a
single-edit mutation usually moves the whole-suite rate by less than the noise
band, so selection is starved and the loop takes almost no effective steps.

This screen restores the gradient. Given a mutation and the reproduction corpus's
verdict on which cases currently fail (the targets) and which pass (the guard), it
applies the diff, runs the gate suite, then benchmarks ONLY those few cases —
seconds instead of the whole suite. A candidate is accepted iff it flips at least
one target red→green AND breaks no guard case. Accepted candidates go on to the
full oracle; rejected ones cost a handful of cases, not a whole run.

Everything dangerous (apply/gates/selective-benchmark) is an injected callable, so
the decision logic is pure and unit-tested with fakes, exactly like
:class:`~misterdev.core.evolution.sandbox.SandboxEvaluator`, which this mirrors —
the difference is only WHICH cases are run and that the verdict is a screen, not a
promotable score.
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple

from misterdev.logging_setup import setup_logger

from .guardrail import ProtectedPathError, assert_mutation_allowed
from .loop import Mutation

logger = setup_logger(__name__)


def _instance_id(result) -> str:
    return str(
        getattr(result, "instance_id", None) or getattr(result, "name", None) or ""
    )


@dataclass
class ScreenVerdict:
    """Outcome of screening one mutation against the targeted + guard cases."""

    accepted: bool
    targeted_total: int
    targeted_resolved: int  # targets that were run and now pass
    guard_total: int
    guard_regressions: int  # guard cases that were run and now fail
    cost: float
    reason: str

    @property
    def rank_key(self) -> Tuple[int, int]:
        """Best-first ordering among survivors: more targets fixed, fewer guard
        breaks (a survivor has zero guard breaks, but this stays correct if the
        acceptance rule is ever relaxed)."""
        return (self.targeted_resolved, -self.guard_regressions)


@dataclass
class MicroEvaluator:
    """Injectable cheap screen: apply -> gates -> run only targeted+guard cases.

    ``target_ids`` are the currently-failing cases a mutation must flip;
    ``guard_ids`` are currently-passing cases it must not break. Both are supplied
    by the reproduction corpus for the run's niche, so the screen measures the
    exact intended effect at a fraction of the oracle's cost.
    """

    apply: Callable[[Mutation], Callable[[], None]]
    gates: Callable[[], bool]
    run_only: Callable[[List[str]], Tuple[list, float]]  # (results, cost)
    target_ids: List[str]
    guard_ids: List[str]

    def screen(self, mutation: Mutation) -> ScreenVerdict:
        """Screen ``mutation``; never raises — a screen failure is a rejection, so
        one bad candidate can never halt the beam."""
        empty = ScreenVerdict(
            accepted=False,
            targeted_total=len(self.target_ids),
            targeted_resolved=0,
            guard_total=len(self.guard_ids),
            guard_regressions=0,
            cost=0.0,
            reason="",
        )
        if not self.target_ids:
            # With nothing to prove, the screen cannot accept — fall through to the
            # oracle by NOT screening (the caller skips the screen when unarmed);
            # if called anyway, reject rather than pass an unmeasured candidate.
            return _with_reason(empty, "no targets to screen against")

        try:
            assert_mutation_allowed(mutation.paths)
        except ProtectedPathError as e:
            return _with_reason(empty, f"guardrail: {e}")

        try:
            teardown = self.apply(mutation)
        except Exception as e:
            logger.warning(f"Screen: apply failed: {e}")
            return _with_reason(empty, f"apply failed: {e}")

        try:
            if not self.gates():
                return _with_reason(empty, "gates failed")
            only = list(dict.fromkeys(self.target_ids + self.guard_ids))
            results, cost = self.run_only(only)
            by_id = {
                _instance_id(r): bool(getattr(r, "resolved", False)) for r in results
            }
            # Count only cases that were actually run: an absent case is unobserved,
            # never counted as a regression (that would reject on a slug mismatch).
            targeted_resolved = sum(1 for t in self.target_ids if by_id.get(t) is True)
            guard_regressions = sum(1 for g in self.guard_ids if by_id.get(g) is False)
            accepted = targeted_resolved > 0 and guard_regressions == 0
            reason = (
                f"fixed {targeted_resolved}/{len(self.target_ids)} targets, "
                f"{guard_regressions} guard regression(s)"
            )
            return ScreenVerdict(
                accepted=accepted,
                targeted_total=len(self.target_ids),
                targeted_resolved=targeted_resolved,
                guard_total=len(self.guard_ids),
                guard_regressions=guard_regressions,
                cost=cost,
                reason=reason,
            )
        except Exception as e:  # selective benchmark failure -> reject, don't crash
            logger.warning(f"Screen: evaluation failed: {e}")
            return _with_reason(empty, f"evaluation failed: {e}")
        finally:
            try:
                teardown()
            except Exception as e:  # cleanup must never mask the verdict
                logger.warning(f"Screen teardown failed (non-fatal): {e}")


def _with_reason(v: ScreenVerdict, reason: str) -> ScreenVerdict:
    return ScreenVerdict(
        accepted=v.accepted,
        targeted_total=v.targeted_total,
        targeted_resolved=v.targeted_resolved,
        guard_total=v.guard_total,
        guard_regressions=v.guard_regressions,
        cost=v.cost,
        reason=reason,
    )
