"""Paired evaluation — resolve a real A-vs-B delta the aggregate rate can't.

Comparing two aggregate resolved-rates (champion vs candidate) throws away the
correlation between them: both scaffolds usually pass/fail the SAME easy and hard
tasks, so the informative signal lives entirely in the DISCORDANT pairs — tasks
one solves and the other doesn't. Scoring those per-task differences (McNemar)
collapses the shared-variance term, so a small run can detect a delta an unpaired
comparison drowns in noise (the measured failure: ~±20% wobble at n=10 made
scaffold deltas unmeasurable). The concordant pairs — both pass or both fail —
carry no signal and are correctly ignored.

Counts can be ACCUMULATED across runs (or read from the reproduction corpus'
per-case history) before deciding, which is how a small-but-real gain earns its
significance over time — fitness by passive accumulation rather than one costly
big run.

Pure and offline: nothing here runs a benchmark. Inputs are per-task pass/fail
maps (or accumulated discordant counts) for two conditions on the SAME task set.
This is a MEASUREMENT primitive; do not optimize a partial-credit score directly
as an RL reward (it is non-monotonic with correctness — see docs/research-
directions.md).
"""

import math
from dataclasses import dataclass
from typing import Mapping, Tuple


@dataclass(frozen=True)
class PairedVerdict:
    """Whether the candidate beats the champion by paired evidence, with the why."""

    promote: bool
    reason: str
    wins: int  # tasks the candidate solves that the champion does not
    losses: int  # tasks the champion solves that the candidate does not (regressions)


def discordant(
    champion: Mapping[str, bool], candidate: Mapping[str, bool]
) -> Tuple[int, int]:
    """(wins, losses) over the tasks BOTH conditions ran. A win: candidate passes
    where champion fails; a loss: champion passes where candidate fails. Concordant
    pairs (agree) are omitted — they carry no comparative signal."""
    shared = champion.keys() & candidate.keys()
    wins = sum(1 for t in shared if candidate[t] and not champion[t])
    losses = sum(1 for t in shared if champion[t] and not candidate[t])
    return wins, losses


def _upper_binom_tail(k: int, n: int) -> float:
    """One-sided P(X >= k) for X ~ Binomial(n, 0.5) — the exact McNemar/sign-test
    tail on the discordant pairs under the null 'no difference'."""
    if n <= 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n)


def decide_paired(wins: int, losses: int, alpha: float = 0.05) -> PairedVerdict:
    """Exact McNemar decision on discordant counts.

    Promote when the candidate wins more than it loses beyond chance (one-sided
    exact binomial p <= alpha over the discordant pairs). A tie, an insignificant
    lead, or a net loss does not promote — and a *significant* net loss is named as
    a regression so the caller can reject it outright. With zero losses this
    reduces to the sign test (needs >= 5 clean wins at alpha=0.05), which is the
    honest bar: 3 lucky wins on one run are not yet distinguishable from noise, but
    they accumulate toward significance across runs.
    """
    n = wins + losses
    if n == 0:
        return PairedVerdict(False, "no discordant pairs — no paired evidence", 0, 0)
    if wins > losses:
        p = _upper_binom_tail(wins, n)
        if p <= alpha:
            return PairedVerdict(
                True,
                f"candidate better: {wins} win(s) vs {losses} (p={p:.3f})",
                wins,
                losses,
            )
        return PairedVerdict(
            False,
            f"gain not significant: {wins} vs {losses} (p={p:.3f} > {alpha})",
            wins,
            losses,
        )
    if losses > wins:
        p = _upper_binom_tail(losses, n)
        tag = "REGRESSION (significant)" if p <= alpha else "worse, not significant"
        return PairedVerdict(
            False,
            f"candidate {tag}: {wins} win(s) vs {losses} (p={p:.3f})",
            wins,
            losses,
        )
    return PairedVerdict(False, f"tie: {wins} win(s) vs {losses}", wins, losses)


def paired_promote(
    champion: Mapping[str, bool], candidate: Mapping[str, bool], alpha: float = 0.05
) -> PairedVerdict:
    """Paired verdict from per-task pass/fail maps for one A/B comparison."""
    wins, losses = discordant(champion, candidate)
    return decide_paired(wins, losses, alpha)


@dataclass(frozen=True)
class PairedPromotion:
    """The anti-overfit gate in paired form: a candidate is promoted only when it
    IMPROVES the derive split by paired evidence AND does not REGRESS the held-out
    split (any significant holdout loss = overfit)."""

    promote: bool
    reason: str


def decide_promotion_paired(
    derive_champ: Mapping[str, bool],
    derive_cand: Mapping[str, bool],
    holdout_champ: Mapping[str, bool],
    holdout_cand: Mapping[str, bool],
    alpha: float = 0.05,
) -> PairedPromotion:
    """Paired analog of :func:`holdout.decide_promotion`.

    Promote iff (a) the candidate significantly improves the DERIVE split by the
    paired test, and (b) it does not significantly regress the HELD-OUT split — a
    holdout loss that clears significance is rejected AS overfit, exactly the
    ratchet the aggregate gate enforces, but with the variance-collapsing power of
    pairing so a real gain is not lost in noise. Holdout merely holding (no net
    change) is fine; the fix's class may simply not appear there.
    """
    d = decide_paired(*discordant(derive_champ, derive_cand), alpha=alpha)
    if not d.promote:
        return PairedPromotion(False, f"derive: {d.reason}")
    h_wins, h_losses = discordant(holdout_champ, holdout_cand)
    h = decide_paired(h_wins, h_losses, alpha=alpha)
    # A significant holdout regression is the overfit signal.
    if h_losses > h_wins and _upper_binom_tail(h_losses, h_wins + h_losses) <= alpha:
        return PairedPromotion(
            False,
            f"OVERFIT: derive gained ({d.reason}) but holdout regressed ({h.reason})",
        )
    return PairedPromotion(
        True, f"generalizes — derive: {d.reason}; holdout: {h.reason}"
    )
