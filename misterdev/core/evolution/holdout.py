"""The held-out generalization gate — the anti-overfit ratchet.

The keep-if-better gate in :mod:`.fitness` compares a candidate to the champion
on ONE task set. That blocks *regressions* but not *overfitting*: a mutation that
lifts the very tasks the proposer was shown while quietly hurting others still
looks like a win when both are measured on the same pool. This module splits the
tasks into disjoint pools and promotes a mutation only when it improves the pool
the proposer *saw* (DERIVE) **without** dropping the pool it *never* saw (HOLDOUT).
A change that trades general capability for derive-specific gain shows up as a
holdout drop and is rejected AS overfit — which is what lets the loop climb the
benchmark without becoming a benchmark-specialist.

The split is by a stable per-slug hash, not a positional shuffle, so a task's pool
membership never flips run-to-run (a task drifting between pools would leak the
holdout). Cross-*distribution* held-out (the strongest form) is achieved by
populating HOLDOUT from a different task source; the gate logic is identical.

Pure and offline-testable: nothing here runs a benchmark or edits code.
"""

import hashlib
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .fitness import FitnessScore


def _bucket(slug: str, seed: int) -> int:
    """A stable 0-999 bucket for a slug (process-independent, unlike hash())."""
    digest = hashlib.sha256(f"{seed}:{slug}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 1000


def split_tasks(
    slugs: Sequence[str], holdout_fraction: float = 0.3, seed: int = 0
) -> Tuple[List[str], List[str]]:
    """Partition ``slugs`` into (derive, holdout), disjoint and stable per slug.

    ``holdout_fraction`` is the target share routed to HOLDOUT. Membership is by a
    stable hash of the slug, so adding/removing tasks never reshuffles the rest and
    a task is always in the same pool. Degenerate inputs (0/1 task) put everything
    in DERIVE so the loop still runs (with no generalization signal, reported by
    the caller) rather than crashing.
    """
    unique = sorted(set(slugs))
    if len(unique) < 2 or holdout_fraction <= 0:
        return list(unique), []
    cutoff = int(max(0.0, min(1.0, holdout_fraction)) * 1000)
    holdout = [s for s in unique if _bucket(s, seed) < cutoff]
    derive = [s for s in unique if _bucket(s, seed) >= cutoff]
    # Guarantee both pools are non-empty when there are ≥2 tasks: if the hash
    # happened to route all to one side, move the single boundary task over.
    if not holdout:
        holdout.append(derive.pop())
    elif not derive:
        derive.append(holdout.pop())
    return derive, holdout


@dataclass(frozen=True)
class PromotionDecision:
    """Whether a mutation generalizes well enough to promote, with the reason."""

    promote: bool
    reason: str


def decide_promotion(
    derive: FitnessScore,
    derive_base: FitnessScore,
    holdout: FitnessScore,
    holdout_base: FitnessScore,
    noise_band: float,
) -> PromotionDecision:
    """The anti-overfit gate: promote iff the mutation gains on DERIVE past noise
    and does not drop HOLDOUT beyond noise, with zero regressions on either pool.

    * Any regression (a baseline-passing task now failing) on either pool → reject;
      correctness is never traded (mirrors :meth:`FitnessScore.beats`).
    * No real gain on DERIVE (delta within the noise band) → reject; the mutation
      did not do its job.
    * HOLDOUT drops beyond the noise band while DERIVE gained → reject AS OVERFIT;
      the change bought derive-specific score at the cost of general capability.
    * Otherwise → promote. HOLDOUT merely holding (neutral) is acceptable: the fixed
      class may not be exercised in HOLDOUT, which is different from harming it.
    """
    if derive.regressions > 0 or holdout.regressions > 0:
        return PromotionDecision(False, "regression on a baseline-passing task")
    d_delta = derive.resolved_rate - derive_base.resolved_rate
    h_delta = holdout.resolved_rate - holdout_base.resolved_rate
    if d_delta <= noise_band:
        return PromotionDecision(
            False, f"no real gain on derive ({d_delta:+.1%}, within noise)"
        )
    if h_delta < -noise_band:
        return PromotionDecision(
            False,
            f"OVERFIT: derive {d_delta:+.1%} but holdout {h_delta:+.1%} — "
            "gain does not generalize, rejected",
        )
    return PromotionDecision(
        True, f"generalizes: derive {d_delta:+.1%}, holdout {h_delta:+.1%}"
    )
