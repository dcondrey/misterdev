import tempfile
from pathlib import Path

import pytest

from misterdev.core.economics.model_ledger import ModelLedger
from misterdev.core.economics.model_selector import ModelSelector


@pytest.fixture
def ledger():
    with tempfile.TemporaryDirectory() as d:
        yield ModelLedger(Path(d) / "model_stats.json")


def _config(**llm):
    base = {
        "dynamic_selection": True,
        "escalation": ["cheap", "strong"],
        "models": {"cheap": "free/x", "strong": "anthropic/big"},
        "selection_posture": "conservative",
        "min_observations": 3,
        "first_try_floor": 0.5,
    }
    base.update(llm)
    return {"llm": base}


def test_disabled_returns_none(ledger):
    sel = ModelSelector(_config(dynamic_selection=False), ledger)
    assert sel.select("feature", "medium", 0, 3) is None


def test_no_escalation_returns_none(ledger):
    sel = ModelSelector(_config(escalation=[]), ledger)
    assert sel.select("feature", "medium", 0, 3) is None


def test_final_attempt_uses_strongest(ledger):
    sel = ModelSelector(_config(), ledger)
    # max_attempts=3 -> attempt 2 is final.
    assert sel.select("feature", "medium", 2, 3) == "anthropic/big"


def test_conservative_first_attempt_falls_back_to_strong_when_unproven(ledger):
    sel = ModelSelector(_config(), ledger)
    # Empty ledger: no cheap model is proven, so the first attempt is strong.
    assert sel.select("feature", "medium", 0, 3) == "anthropic/big"


def test_conservative_first_attempt_uses_proven_cheap(ledger):
    # Make the cheap model proven for first-try on this context.
    for _ in range(3):
        ledger.record(
            "free/x", "feature", "medium", success=True, first_try=True, cost=0.001
        )
    sel = ModelSelector(_config(), ledger)
    assert sel.select("feature", "medium", 0, 3) == "free/x"


def test_proven_requires_floor(ledger):
    # 3 first-try attempts but only 1 success -> below 0.5 floor, not trusted.
    ledger.record("free/x", "feature", "medium", success=True, first_try=True)
    ledger.record("free/x", "feature", "medium", success=False, first_try=True)
    ledger.record("free/x", "feature", "medium", success=False, first_try=True)
    sel = ModelSelector(_config(), ledger)
    assert sel.select("feature", "medium", 0, 3) == "anthropic/big"


def test_climbing_to_strong_on_retry(ledger):
    for _ in range(3):
        ledger.record(
            "free/x", "feature", "medium", success=True, first_try=True, cost=0.001
        )
    sel = ModelSelector(_config(), ledger)
    # attempt 0 -> proven cheap; attempt 1 (not final, max=3) climbs one rung.
    assert sel.select("feature", "medium", 0, 3) == "free/x"
    assert sel.select("feature", "medium", 1, 3) == "anthropic/big"


def test_aggressive_explores_cheap_first(ledger):
    sel = ModelSelector(_config(selection_posture="aggressive"), ledger)
    # Even with an empty ledger, aggressive tries the cheapest tier first
    # (unseen -> +inf UCB).
    assert sel.select("feature", "medium", 0, 3) == "free/x"


def test_balanced_explores_low_complexity_but_not_high(ledger):
    sel = ModelSelector(_config(selection_posture="balanced"), ledger)
    assert sel.select("feature", "small", 0, 3) == "free/x"
    # High complexity stays conservative -> strong when nothing proven.
    assert sel.select("feature", "large", 0, 3) == "anthropic/big"


def test_auto_string_enables_selection(ledger):
    sel = ModelSelector(_config(dynamic_selection="auto"), ledger)
    assert sel.enabled is True
    assert sel.auto is True


def test_non_auto_string_is_off(ledger):
    # A stray string like "false" must not be treated as truthy-enabled.
    sel = ModelSelector(_config(dynamic_selection="false"), ledger)
    assert sel.enabled is False
    assert sel.select("feature", "medium", 0, 3) is None


def test_auto_warmup_explores_easy_tasks(ledger):
    sel = ModelSelector(
        _config(dynamic_selection="auto", maturity_threshold=10), ledger
    )
    # Immature cell + low complexity -> explore the cheap tier first.
    assert sel.select("feature", "small", 0, 3) == "free/x"


def test_filtered_tier_climbs_to_next_rung(ledger):
    # When the cheap tier is emptied by a filter (here: proven incompetent), the
    # selector climbs to the stronger tier rather than returning None (default).
    cfg = _config(selection_posture="aggressive")
    for _ in range(4):
        ledger.record("free/x", "feature", "medium", success=False)
    sel = ModelSelector(cfg, ledger)
    assert sel.select("feature", "medium", 0, 3) == "anthropic/big"


def test_warm_start_borrows_proven_performance_from_other_cells(ledger):
    # A model proven on (feature, small) is trusted on a COLD (feature, medium)
    # cell before it has local data there — shrinkage toward its global rate.
    cfg = _config()
    for _ in range(4):
        ledger.record(
            "free/x", "feature", "small", success=True, first_try=True, cost=0.001
        )
    sel = ModelSelector(cfg, ledger)
    assert sel._proven("free/x", "feature", "medium") is True
    assert sel.select("feature", "medium", 0, 3) == "free/x"


def test_warm_start_does_not_trust_a_globally_weak_model(ledger):
    # A model that fails everywhere is NOT warm-started into a new cell.
    cfg = _config()
    for _ in range(4):
        ledger.record("free/x", "feature", "small", success=False, first_try=True)
    sel = ModelSelector(cfg, ledger)
    assert sel._proven("free/x", "feature", "medium") is False
    assert sel.select("feature", "medium", 0, 3) == "anthropic/big"


def test_warm_start_prior_yields_to_local_evidence(ledger):
    # The prior is small: a few real failures on the cell override a strong global
    # rate, so a model that can't do THIS cell stops being trusted here.
    cfg = _config()
    for _ in range(6):
        ledger.record(
            "free/x", "feature", "small", success=True, first_try=True, cost=0.001
        )
    for _ in range(5):
        ledger.record("free/x", "feature", "medium", success=False, first_try=True)
    sel = ModelSelector(cfg, ledger)
    # global is strong but local (medium) evidence of failure dominates.
    assert sel._proven("free/x", "feature", "medium") is False


def test_incompetent_model_is_avoided(ledger):
    # A model with enough attempts and a success rate below the floor is proven
    # unable to do this kind of task and must not be selected for it.
    cfg = _config(selection_posture="aggressive")
    for _ in range(4):
        ledger.record("free/x", "feature", "small", success=False)
    sel = ModelSelector(cfg, ledger)
    assert sel._incompetent("free/x", "feature", "small") is True
    # aggressive would normally explore the cheap model first; the guard excludes it.
    assert sel.select("feature", "small", 0, 3) != "free/x"
    # the strongest tier is still reachable on the final attempt.
    assert sel.select("feature", "small", 2, 3) == "anthropic/big"


def test_slow_model_excluded_on_nonfinal_attempts(ledger):
    # A model proven too slow for the wall-clock budget is skipped on non-final
    # attempts; the final attempt lifts the latency guard (capable last resort).
    cfg = _config(selection_posture="aggressive", max_attempt_latency_seconds=100)
    for _ in range(3):
        ledger.record(
            "free/x", "feature", "small", success=True, first_try=True, latency=300.0
        )
    sel = ModelSelector(cfg, ledger)
    assert sel._too_slow("free/x", "feature", "small") is True
    assert sel.select("feature", "small", 0, 3) != "free/x"  # non-final: skipped
    # final attempt uses the strongest tier regardless.
    assert sel.select("feature", "small", 2, 3) == "anthropic/big"


def test_free_endpoints_reserved_for_easy_tasks(ledger):
    # `:free` models are slow/unreliable, so they're used only on trivial/small.
    cfg = _config(
        models={"cheap": "x/model:free", "strong": "anthropic/big"},
        selection_posture="aggressive",
    )
    sel = ModelSelector(cfg, ledger)
    # small: the free endpoint is explored.
    assert sel.select("feature", "small", 0, 3) == "x/model:free"
    # medium: the free endpoint is skipped (never selected).
    assert sel.select("feature", "medium", 0, 3) != "x/model:free"
    # final attempt on medium climbs to the paid model, never free.
    assert sel.select("feature", "medium", 2, 3) == "anthropic/big"


def test_auto_does_not_explore_unproven_on_medium(ledger):
    # Auto mode must NOT gamble a first attempt on an unproven cheap model for a
    # medium task: on a large repo those returned no usable edits. It goes strong.
    sel = ModelSelector(
        _config(dynamic_selection="auto", maturity_threshold=10), ledger
    )
    assert sel.select("feature", "medium", 0, 3) == "anthropic/big"


def test_auto_still_uses_proven_cheap_on_medium(ledger):
    # A cheap model that has EARNED trust is still used first on medium.
    for _ in range(3):
        ledger.record(
            "free/x", "feature", "medium", success=True, first_try=True, cost=0.001
        )
    sel = ModelSelector(
        _config(dynamic_selection="auto", maturity_threshold=10), ledger
    )
    assert sel.select("feature", "medium", 0, 3) == "free/x"


def test_auto_warmup_keeps_hard_tasks_on_strong(ledger):
    sel = ModelSelector(
        _config(dynamic_selection="auto", maturity_threshold=10), ledger
    )
    assert sel.select("feature", "large", 0, 3) == "anthropic/big"


def test_auto_matures_into_conservative(ledger):
    cfg = _config(dynamic_selection="auto", maturity_threshold=3)
    # Fill the cell with cheap-model failures: it matures but nothing is proven,
    # so it settles back to the strong model on the first attempt.
    for _ in range(3):
        ledger.record("free/x", "feature", "small", success=False, first_try=True)
    sel = ModelSelector(cfg, ledger)
    assert sel.select("feature", "small", 0, 3) == "anthropic/big"


def test_auto_uses_proven_cheap_after_maturing(ledger):
    cfg = _config(dynamic_selection="auto", maturity_threshold=3, min_observations=3)
    for _ in range(3):
        ledger.record(
            "free/x", "feature", "small", success=True, first_try=True, cost=0.001
        )
    sel = ModelSelector(cfg, ledger)
    assert sel.select("feature", "small", 0, 3) == "free/x"


def test_auto_self_assembles_ladder_from_free_models(ledger):
    # No escalation/models configured at all: just a default model + free list.
    config = {
        "llm": {
            "dynamic_selection": "auto",
            "model": "anthropic/default",
            "maturity_threshold": 10,
        }
    }
    sel = ModelSelector(config, ledger, free_models=["vendor/free:free"])
    assert sel.escalation == ["_free", "_default"]
    # Easy task in warmup -> tries the free model first; final attempt -> default.
    assert sel.select("feature", "small", 0, 3) == "vendor/free:free"
    assert sel.select("feature", "small", 2, 3) == "anthropic/default"


def test_picks_cheaper_when_quality_per_dollar_higher(ledger):
    # Two candidates in the cheap tier; the cheaper-but-equally-good one wins.
    cfg = _config(
        models={"cheap": ["free/a", "free/b"], "strong": "anthropic/big"},
        selection_posture="aggressive",
    )
    for _ in range(10):
        ledger.record(
            "free/a", "feature", "medium", success=True, first_try=True, cost=0.01
        )
        ledger.record(
            "free/b", "feature", "medium", success=True, first_try=True, cost=0.10
        )
    sel = ModelSelector(cfg, ledger)
    # Equal quality, free/a is 10x cheaper -> higher quality-per-dollar.
    assert sel.select("feature", "medium", 0, 3) == "free/a"
