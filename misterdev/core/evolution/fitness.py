"""The fitness function — the objective every self-edit is judged against.

This is the heart of the empirical self-improvement loop (AlphaEvolve / Darwin-
Gödel style): a machine-gradeable score derived from the benchmark harness, plus
the *keep-if-better-beyond-noise* decision that stops the loop from adopting a
change that merely rode benchmark variance. Nothing here runs a benchmark or
edits code; it only turns a benchmark READING into a comparable score, so it is
pure and fully testable without spending a cent.

Objectives (multi-objective, lexicographic behind a noise band):

1. ``regressions == 0`` — a HARD gate; any task that passed at baseline and now
   fails disqualifies the candidate outright, no matter how the rate moved.
2. ``resolved_rate`` ↑ — the primary objective. A gain must exceed the harness's
   measured noise band to count; a smaller gain is indistinguishable from noise.
3. ``cost_per_task`` ↓ — the tie-breaker when resolved_rate is a statistical tie.

Reads are duck-typed on the evaluation ``SuiteReport`` (``.resolved`` / ``.total``)
so this module imports nothing from ``evaluation`` and the layering stays one-way.
"""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class FitnessScore:
    """One benchmark run's outcome, reduced to the comparable objectives.

    Immutable so a score archived as an incumbent can never be mutated out from
    under a later comparison.
    """

    resolved: int
    total: int
    cost: float  # total dollars for the run
    regressions: int = 0  # tasks that PASSED at baseline and now FAIL

    @property
    def resolved_rate(self) -> float:
        return self.resolved / self.total if self.total else 0.0

    @property
    def cost_per_task(self) -> float:
        return self.cost / self.total if self.total else 0.0

    @classmethod
    def from_report(cls, report, cost: float, regressions: int = 0) -> "FitnessScore":
        """Read a duck-typed benchmark report (``.resolved``/``.total`` ints) plus
        the run's measured cost into a score. ``regressions`` is supplied by the
        caller, which alone knows the baseline pass set."""
        return cls(
            resolved=int(getattr(report, "resolved", 0)),
            total=int(getattr(report, "total", 0)),
            cost=float(cost),
            regressions=int(regressions),
        )

    def beats(self, incumbent: "FitnessScore", noise_band: float) -> bool:
        """True iff this score is a REAL improvement over ``incumbent``.

        The keep-if-better rule, in order:

        * Any regression disqualifies (hard gate) — correctness is never traded
          for rate or cost.
        * A resolved-rate gain must exceed ``noise_band`` (absolute, on the rate)
          to count; a delta inside the band is a statistical tie, not a win.
        * On a quality tie, a strictly cheaper run wins (equal capability for less
          money is a real, adopt-worthy improvement); otherwise it does not beat.
        """
        if self.regressions > 0:
            return False
        delta = self.resolved_rate - incumbent.resolved_rate
        if delta > noise_band:
            return True
        if delta < -noise_band:
            return False
        # Quality tie within the noise band: decide on cost, strictly cheaper.
        return self.cost_per_task < incumbent.cost_per_task


def estimate_noise_band(rates: Sequence[float]) -> float:
    """The noise band from repeated runs of the SAME configuration.

    Returns the population standard deviation of the resolved-rates — the spread
    within which a delta is indistinguishable from run-to-run harness noise, so a
    candidate must beat it to be believed. Fewer than two samples yields ``0.0``
    (no evidence of noise); callers should fall back to a conservative default
    band rather than trusting a zero.
    """
    n = len(rates)
    if n < 2:
        return 0.0
    mean = sum(rates) / n
    variance = sum((r - mean) ** 2 for r in rates) / n
    return variance**0.5
