import json
import tempfile
from pathlib import Path

import pytest

from my_project_orchestrator.core.model_ledger import ModelLedger, ModelStat


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
