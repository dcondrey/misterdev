import json
import tempfile
from pathlib import Path

import pytest

from misterdev.core.economics.model_ledger import ModelLedger, ModelStat


@pytest.fixture
def ledger_path():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d) / "model_stats.json"


def test_record_creates_and_persists(ledger_path):
    ledger = ModelLedger(ledger_path)
    ledger.record(
        "cheap/model",
        "feature",
        "medium",
        success=True,
        first_try=True,
        cost=0.01,
        latency=2.0,
        timestamp=100.0,
    )
    assert ledger_path.exists()
    reloaded = ModelLedger(ledger_path)
    s = reloaded.stat("cheap/model", "feature", "medium")
    assert s.attempts == 1
    assert s.successes == 1
    assert s.first_try_successes == 1
    assert s.total_cost == pytest.approx(0.01)
    assert s.last_seen == 100.0


def test_cost_only_accrues_on_success(ledger_path):
    ledger = ModelLedger(ledger_path)
    ledger.record("m", "c", "x", success=False, cost=0.05)
    s = ledger.stat("m", "c", "x")
    assert s.attempts == 1
    assert s.successes == 0
    assert s.total_cost == 0.0
    assert s.avg_cost == 0.0


def test_rates(ledger_path):
    ledger = ModelLedger(ledger_path)
    for ok in (True, True, False, True):
        ledger.record("m", "c", "x", success=ok, first_try=True)
    ledger.record("m", "c", "x", success=False, aborted=True)
    s = ledger.stat("m", "c", "x")
    assert s.success_rate == pytest.approx(3 / 5)
    assert s.first_try_rate == pytest.approx(3 / 4)
    assert s.abort_rate == pytest.approx(1 / 5)


def test_quality_score_penalizes_undersampling():
    lucky = ModelStat(model="a", attempts=1, successes=1)
    proven = ModelStat(model="b", attempts=100, successes=95)
    # Both have high raw rates, but the 1/1 model's Wilson lower bound is much
    # lower because the interval is wide.
    assert proven.quality_score() > lucky.quality_score()
    assert lucky.quality_score() < 1.0


def test_quality_score_zero_when_unseen():
    assert ModelStat(model="a").quality_score() == 0.0


def test_quality_score_clamps_inconsistent_counts():
    # A torn read (or float drift) can make successes exceed attempts; the score
    # must clamp p to [0,1] and return a valid bound instead of raising
    # ValueError from sqrt of a negative.
    torn = ModelStat(model="a", attempts=2.0, successes=3.0)
    score = torn.quality_score()
    assert 0.0 <= score <= 1.0


def test_stat_returns_independent_snapshot(ledger_path):
    ledger = ModelLedger(ledger_path)
    ledger.record("m", "feature", "medium", success=True, timestamp=100.0)
    snap = ledger.stat("m", "feature", "medium")
    snap.attempts += 999  # mutate the returned copy
    # The ledger's own cell is untouched by mutating the snapshot.
    assert ledger.stat("m", "feature", "medium").attempts == 1.0


def test_selection_score_explores_unseen_model(ledger_path):
    ledger = ModelLedger(ledger_path)
    # A proven model.
    for _ in range(20):
        ledger.record("proven", "c", "x", success=True)
    total = ledger.total_observations("c", "x")
    seen = ledger.selection_score("proven", "c", "x", total)
    unseen = ledger.selection_score("brand-new", "c", "x", total)
    # An unseen model gets +inf priority so it is always tried at least once.
    assert unseen == float("inf")
    assert seen < float("inf")


def test_selection_score_exploration_bonus_shrinks_with_attempts(ledger_path):
    ledger = ModelLedger(ledger_path)
    for _ in range(5):
        ledger.record("a", "c", "x", success=True)
    for _ in range(50):
        ledger.record("b", "c", "x", success=True)
    total = ledger.total_observations("c", "x")
    # Same perfect success rate, but the less-sampled model carries a larger
    # exploration bonus, so it scores higher (still worth exploring).
    assert ledger.selection_score("a", "c", "x", total) > ledger.selection_score(
        "b", "c", "x", total
    )


def test_corrupt_file_starts_fresh(ledger_path):
    ledger_path.write_text("{not valid json", encoding="utf-8")
    ledger = ModelLedger(ledger_path)
    assert ledger.known_models() == []
    # And it can still record afterward.
    ledger.record("m", "c", "x", success=True)
    assert "m" in ledger.known_models()


def test_unknown_keys_dropped_on_load(ledger_path):
    ledger_path.write_text(
        json.dumps(
            {
                "m␟c␟x": {
                    "model": "m",
                    "category": "c",
                    "complexity": "x",
                    "attempts": 3,
                    "successes": 2,
                    "obsolete_field": 9,
                }
            }
        ),
        encoding="utf-8",
    )
    ledger = ModelLedger(ledger_path)
    s = ledger.stat("m", "c", "x")
    assert s.attempts == 3
    assert s.successes == 2


def test_context_scoped_keys_are_distinct(ledger_path):
    ledger = ModelLedger(ledger_path)
    ledger.record("m", "feature", "large", success=True)
    ledger.record("m", "test", "small", success=False)
    assert ledger.stat("m", "feature", "large").successes == 1
    assert ledger.stat("m", "test", "small").successes == 0


# --- recency decay ----------------------------------------------------------


def test_decay_factor_dead_band_and_half_life():
    from misterdev.core.economics.model_ledger import (
        _decay_factor,
        _DEFAULT_HALF_LIFE_SECONDS,
    )

    hl = _DEFAULT_HALF_LIFE_SECONDS
    # No time / sub-hour / backward clock / disabled -> exactly 1.0.
    assert _decay_factor(0.0, hl) == 1.0
    assert _decay_factor(60.0, hl) == 1.0
    assert _decay_factor(-5.0, hl) == 1.0
    assert _decay_factor(10 * 86400, 0) == 1.0
    # One half-life elapsed -> ~0.5; two -> ~0.25.
    assert abs(_decay_factor(hl, hl) - 0.5) < 1e-9
    assert abs(_decay_factor(2 * hl, hl) - 0.25) < 1e-9


def test_within_build_records_are_not_decayed(ledger_path):
    # Several attempts seconds apart (a single build) stay whole integers, so
    # the selector's integer sample thresholds aren't tripped by tiny drift.
    ledger = ModelLedger(ledger_path)
    for i in range(3):
        ledger.record("m", "feature", "medium", success=True, timestamp=1000.0 + i)
    s = ledger.stat("m", "feature", "medium")
    assert s.attempts == 3
    assert s.successes == 3


def test_stale_outcomes_decay_on_new_record(ledger_path):
    from misterdev.core.economics.model_ledger import (
        _DEFAULT_HALF_LIFE_SECONDS,
    )

    ledger = ModelLedger(ledger_path)
    # Two successes far in the past, then one fresh success a half-life later.
    ledger.record("m", "feature", "medium", success=True, timestamp=0.0)
    ledger.record("m", "feature", "medium", success=True, timestamp=1.0)
    s_before = ledger.stat("m", "feature", "medium")
    assert s_before.attempts == 2
    ledger.record(
        "m", "feature", "medium", success=True, timestamp=_DEFAULT_HALF_LIFE_SECONDS
    )
    s = ledger.stat("m", "feature", "medium")
    # Prior 2 decayed by ~0.5 -> ~1.0, plus the new one -> ~2.0 effective.
    assert 1.9 < s.attempts < 2.1
    assert 1.9 < s.successes < 2.1


def test_decay_preserves_success_rate(ledger_path):
    from misterdev.core.economics.model_ledger import (
        _DEFAULT_HALF_LIFE_SECONDS,
    )

    ledger = ModelLedger(ledger_path)
    ledger.record("m", "feature", "medium", success=True, timestamp=0.0)
    ledger.record("m", "feature", "medium", success=False, timestamp=1.0)
    # A later record decays both numerator and denominator equally.
    ledger.record(
        "m", "feature", "medium", success=False, timestamp=_DEFAULT_HALF_LIFE_SECONDS
    )
    s = ledger.stat("m", "feature", "medium")
    # 1 success out of ~3 effective attempts -> rate near 1/3 (was 1/2 of 2 old
    # plus a fresh failure), proving the ratio tracks recent outcomes.
    assert 0.0 < s.success_rate < 0.5


def test_decay_disabled_keeps_raw_counts(ledger_path):
    ledger = ModelLedger(ledger_path, half_life_seconds=0)
    ledger.record("m", "feature", "medium", success=True, timestamp=0.0)
    ledger.record("m", "feature", "medium", success=True, timestamp=10 * 365 * 86400.0)
    assert ledger.stat("m", "feature", "medium").attempts == 2
