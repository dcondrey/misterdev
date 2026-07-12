"""Partial-credit (ordinal) fitness — a lower-variance MEASUREMENT metric.

Binary resolved/total is a Bernoulli signal: a task that compiled, constructed
the type, and passed every mutation check but missed one derivation scores the
SAME zero as a task that never compiled. On a hard benchmark most of the signal
lives in that hidden partial progress, so the pass-rate estimate is both coarse
and high-variance — the loop can't tell a real 2% gain from noise (docs/
research-directions.md, Theme 2).

Grade each task instead by how far it climbed an ORDERED checkpoint ladder
(mirroring the verifier-decomposition stages: compile → construct → invariant →
query → suite). Reaching a later checkpoint implies the earlier ones, so credit
is the fraction of the ladder cleared. The mean over tasks is a denser, finer-
resolution progress signal than pass-rate.

GUARDRAIL (Pass-Rate-Reward null result, 2605.02944): partial credit is
NON-MONOTONIC with correctness and MUST NOT become a training/promotion reward —
optimizing it rewards half-finished work. It is a measurement metric ONLY: use
it to *resolve* "did this change help?" faster; the promotion gate stays
regressions==0 + resolved_rate (see fitness.py). Pure and deterministic.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

# Ordered checkpoints; index+1 over the length is the graded credit for reaching
# that checkpoint. "suite" (all tests green) is full credit == a binary pass.
CHECKPOINTS: Tuple[str, ...] = ("compile", "construct", "invariant", "query", "suite")
_INDEX = {name: i for i, name in enumerate(CHECKPOINTS)}


def checkpoint_credit(reached: Optional[str]) -> float:
    """Graded credit in [0, 1] for the FURTHEST checkpoint a task reached.

    ``None`` / unknown -> 0.0 (did not even compile). ``"suite"`` -> 1.0."""
    if reached is None:
        return 0.0
    idx = _INDEX.get(reached)
    if idx is None:
        return 0.0
    return (idx + 1) / len(CHECKPOINTS)


def furthest_checkpoint(reached: Iterable[str]) -> Optional[str]:
    """The latest checkpoint in an unordered set of reached checkpoints."""
    best = None
    best_idx = -1
    for name in reached:
        idx = _INDEX.get(name, -1)
        if idx > best_idx:
            best_idx, best = idx, name
    return best


@dataclass(frozen=True)
class PartialCreditScore:
    """A run's per-task ordinal credits, reduced to a comparable metric.

    ``credits`` is one value in [0, 1] per task (its furthest checkpoint). The
    mean is the progress signal; ``resolved_rate`` (credit == 1.0) is retained so
    this can be reported alongside the binary metric without replacing it.
    """

    credits: Tuple[float, ...]

    @classmethod
    def from_checkpoints(
        cls, per_task: Sequence[Iterable[str] | str | None]
    ) -> "PartialCreditScore":
        """Build from each task's reached checkpoint(s): a set/list of names, a
        single name, or ``None``."""
        out = []
        for reached in per_task:
            if reached is None or isinstance(reached, str):
                out.append(checkpoint_credit(reached))
            else:
                out.append(checkpoint_credit(furthest_checkpoint(reached)))
        return cls(tuple(out))

    @property
    def total(self) -> int:
        return len(self.credits)

    @property
    def mean_credit(self) -> float:
        return sum(self.credits) / self.total if self.total else 0.0

    @property
    def resolved(self) -> int:
        """Tasks at full credit — identical to the binary resolved count."""
        return sum(1 for c in self.credits if c >= 1.0)

    @property
    def resolved_rate(self) -> float:
        return self.resolved / self.total if self.total else 0.0

    @property
    def variance(self) -> float:
        """Sample variance of the per-task credits (population form)."""
        if self.total < 1:
            return 0.0
        m = self.mean_credit
        return sum((c - m) ** 2 for c in self.credits) / self.total

    @property
    def stderr(self) -> float:
        """Standard error of the mean-credit estimate."""
        if self.total < 2:
            return 0.0
        return (self.variance / self.total) ** 0.5

    def delta(self, incumbent: "PartialCreditScore") -> float:
        """Change in mean credit vs. an incumbent (positive == more progress)."""
        return self.mean_credit - incumbent.mean_credit

    def resolves(self, incumbent: "PartialCreditScore", z: float = 2.0) -> bool:
        """True iff the mean-credit gain clears ``z`` pooled standard errors — the
        finer-grained companion to :meth:`FitnessScore.beats`, for READING whether
        a change moved the needle. Advisory only; never a promotion gate."""
        pooled = (self.stderr**2 + incumbent.stderr**2) ** 0.5
        if pooled == 0.0:
            return self.mean_credit > incumbent.mean_credit
        return self.delta(incumbent) > z * pooled
