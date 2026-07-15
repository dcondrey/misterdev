"""Adaptive concurrency/timeout backoff decision (pure function)."""

from misterdev.core.execution.adaptive import WaveTuning, next_wave_tuning


def test_backoff_on_infra_above_threshold():
    """A wave over the threshold halves workers and grows the timeout factor."""
    nxt = next_wave_tuning(3, WaveTuning(8, 1.0), base_workers=8, threshold=1)
    assert nxt == WaveTuning(4, 2.0)


def test_recover_on_clean_wave():
    """A clean wave (0 infra) steps workers and timeout back toward the base."""
    nxt = next_wave_tuning(0, WaveTuning(2, 4.0), base_workers=8)
    assert nxt == WaveTuning(4, 2.0)


def test_hold_steady_at_or_below_threshold():
    """A nonzero count at/below the threshold neither backs off nor recovers."""
    cur = WaveTuning(4, 2.0)
    assert next_wave_tuning(1, cur, base_workers=8, threshold=1) == cur


def test_worker_floor_respected():
    """Backoff never drops below the worker floor, even from the floor."""
    nxt = next_wave_tuning(9, WaveTuning(1, 2.0), base_workers=8, min_workers=1)
    assert nxt.max_workers == 1


def test_timeout_ceiling_respected():
    """The timeout factor is capped at max_timeout_factor under sustained infra."""
    nxt = next_wave_tuning(
        9, WaveTuning(4, 4.0), base_workers=8, max_timeout_factor=4.0
    )
    assert nxt.timeout_factor == 4.0


def test_recovery_clamps_to_base_workers_and_factor_one():
    """Recovery never overshoots the configured base workers or drops below 1.0."""
    nxt = next_wave_tuning(0, WaveTuning(8, 1.0), base_workers=8)
    assert nxt == WaveTuning(8, 1.0)


def test_backoff_then_recover_round_trip():
    """A bad wave then two clean waves returns to full concurrency and factor 1.0."""
    t = WaveTuning(8, 1.0)
    t = next_wave_tuning(5, t, base_workers=8)  # -> (4, 2.0)
    assert t == WaveTuning(4, 2.0)
    t = next_wave_tuning(0, t, base_workers=8)  # -> (8, 1.0)
    assert t == WaveTuning(8, 1.0)


def test_base_workers_one_stays_one():
    """With a configured cap of 1, workers stay pinned at 1 through backoff and
    recovery; only the timeout factor adapts."""
    t = WaveTuning(1, 1.0)
    t = next_wave_tuning(5, t, base_workers=1)
    assert t.max_workers == 1 and t.timeout_factor == 2.0
    t = next_wave_tuning(0, t, base_workers=1)
    assert t == WaveTuning(1, 1.0)
