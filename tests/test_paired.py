"""Paired evaluation: resolve an A/B delta the aggregate rate can't, by scoring
the discordant pairs (McNemar). Pure/offline."""

from misterdev.core.evolution.paired import (
    decide_paired,
    decide_promotion_paired,
    discordant,
    paired_promote,
)


def test_discordant_counts_ignore_concordant_pairs():
    champ = {"a": True, "b": False, "c": True, "d": False}
    cand = {"a": True, "b": True, "c": False, "d": False}
    # a: both pass (concordant), d: both fail (concordant) -> ignored.
    # b: candidate newly passes -> win. c: champion passed, candidate fails -> loss.
    assert discordant(champ, cand) == (1, 1)


def test_sign_test_threshold_five_clean_wins():
    # Exact sign test at alpha=0.05: 5 clean wins clears (p=1/32), 4 does not (1/16).
    assert decide_paired(5, 0).promote
    assert not decide_paired(4, 0).promote
    assert "p=0.031" in decide_paired(5, 0).reason


def test_a_single_loss_raises_the_bar():
    # 6 wins vs 1 loss is NOT yet significant (p=8/128=0.0625); 7 vs 1 is (9/256).
    assert not decide_paired(6, 1).promote
    assert decide_paired(7, 1).promote


def test_net_loss_named_as_regression():
    v = decide_paired(0, 5)
    assert not v.promote and "REGRESSION" in v.reason


def test_tie_and_no_evidence_do_not_promote():
    assert not decide_paired(3, 3).promote  # tie
    assert not decide_paired(0, 0).promote  # no discordant pairs
    assert "no paired evidence" in decide_paired(0, 0).reason


def test_paired_distinguishes_real_gain_from_churn():
    # Five clean wins, zero losses -> a real gain, promoted.
    champ = {f"t{i}": False for i in range(8)} | {"keep": True}
    gain = {f"t{i}": (i < 5) for i in range(8)} | {"keep": True}
    assert discordant(champ, gain) == (5, 0)
    assert paired_promote(champ, gain).promote

    # Same five newly-passing tasks, but the candidate also breaks five the
    # champion passed: net resolved-count is unchanged, so an aggregate-rate
    # comparison sees zero delta and is blind. The paired test sees 5 wins vs 5
    # losses = churn, and does not promote.
    champ2 = {f"t{i}": False for i in range(8)} | {f"k{j}": True for j in range(5)}
    cand2 = {f"t{i}": (i < 5) for i in range(8)} | {f"k{j}": False for j in range(5)}
    assert discordant(champ2, cand2) == (5, 5)
    assert not paired_promote(champ2, cand2).promote


def test_promotion_paired_promotes_generalizing_gain():
    derive_champ = {f"d{i}": False for i in range(6)}
    derive_cand = {f"d{i}": (i < 5) for i in range(6)}  # 5 clean wins
    holdout_champ = {"h1": True, "h2": True}
    holdout_cand = {"h1": True, "h2": True}  # holdout holds
    v = decide_promotion_paired(derive_champ, derive_cand, holdout_champ, holdout_cand)
    assert v.promote and "generalizes" in v.reason


def test_promotion_paired_rejects_overfit():
    derive_champ = {f"d{i}": False for i in range(6)}
    derive_cand = {f"d{i}": (i < 5) for i in range(6)}  # 5 derive wins
    # ...but the candidate breaks the whole held-out split it used to pass.
    holdout_champ = {f"h{i}": True for i in range(5)}
    holdout_cand = {f"h{i}": False for i in range(5)}  # 5 holdout regressions
    v = decide_promotion_paired(derive_champ, derive_cand, holdout_champ, holdout_cand)
    assert not v.promote and "OVERFIT" in v.reason


def test_promotion_paired_rejects_insignificant_derive_gain():
    derive_champ = {f"d{i}": False for i in range(6)}
    derive_cand = {f"d{i}": (i < 3) for i in range(6)}  # only 3 wins -> not significant
    v = decide_promotion_paired(derive_champ, derive_cand, {}, {})
    assert not v.promote and "derive" in v.reason
