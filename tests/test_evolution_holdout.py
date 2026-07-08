"""The held-out gate splits leak-free and rejects overfitting."""

from misterdev.core.evolution.fitness import FitnessScore
from misterdev.core.evolution.holdout import (
    decide_promotion,
    split_tasks,
)


def _score(resolved, total, cost=1.0, regressions=0):
    return FitnessScore(
        resolved=resolved, total=total, cost=cost, regressions=regressions
    )


def test_split_is_disjoint_and_covers_all():
    slugs = [f"ex{i}" for i in range(20)]
    derive, holdout = split_tasks(slugs, holdout_fraction=0.3)
    assert set(derive).isdisjoint(holdout)
    assert set(derive) | set(holdout) == set(slugs)
    assert holdout and derive  # both non-empty
    assert 0 < len(holdout) < len(slugs)


def test_split_is_stable_per_slug_across_list_changes():
    # A task's pool membership must not flip when the task list changes, or the
    # holdout leaks. Same slug -> same pool, regardless of neighbors.
    base = [f"ex{i}" for i in range(20)]
    _, holdout_a = split_tasks(base, holdout_fraction=0.3, seed=7)
    grown = base + [f"new{i}" for i in range(10)]
    _, holdout_b = split_tasks(grown, holdout_fraction=0.3, seed=7)
    # Every original slug keeps its holdout membership after the list grew.
    assert {s for s in holdout_a} == {s for s in holdout_b if s in base}


def test_split_degenerate_inputs_all_derive():
    assert split_tasks([], 0.3) == ([], [])
    assert split_tasks(["only"], 0.3) == (["only"], [])


def test_promote_when_gain_generalizes():
    d = decide_promotion(
        derive=_score(8, 10),
        derive_base=_score(6, 10),
        holdout=_score(7, 10),
        holdout_base=_score(6, 10),
        noise_band=0.05,
    )
    assert d.promote and "generalizes" in d.reason


def test_reject_overfit_derive_up_holdout_down():
    # The signature of overfitting: derive rises, holdout falls beyond noise.
    d = decide_promotion(
        derive=_score(9, 10),
        derive_base=_score(6, 10),
        holdout=_score(4, 10),
        holdout_base=_score(6, 10),
        noise_band=0.05,
    )
    assert not d.promote and "OVERFIT" in d.reason


def test_accept_when_holdout_merely_holds():
    # Neutral holdout (class not exercised there) is NOT overfit — accept.
    d = decide_promotion(
        derive=_score(8, 10),
        derive_base=_score(6, 10),
        holdout=_score(6, 10),
        holdout_base=_score(6, 10),
        noise_band=0.05,
    )
    assert d.promote


def test_reject_when_no_real_derive_gain():
    d = decide_promotion(
        derive=_score(6, 10),
        derive_base=_score(6, 10),
        holdout=_score(9, 10),
        holdout_base=_score(6, 10),
        noise_band=0.05,
    )
    assert not d.promote and "no real gain" in d.reason


def test_regression_hard_rejects_even_with_gains():
    d = decide_promotion(
        derive=_score(9, 10, regressions=1),
        derive_base=_score(6, 10),
        holdout=_score(9, 10),
        holdout_base=_score(6, 10),
        noise_band=0.05,
    )
    assert not d.promote and "regression" in d.reason
