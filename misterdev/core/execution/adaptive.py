"""Adaptive concurrency/timeout backoff on infrastructure faults.

When a wave shows repeated ENVIRONMENT faults (see ``infra.py``) — gate timeouts,
locked package stores, OOM — the machine is contended, not the code broken.
Thrashing on at full concurrency makes it worse: every parallel gate competes for
the same CPU/store and times out. The right response is to back off — fewer
concurrent workers, longer timeouts — for the NEXT wave, then recover gradually
once waves come back clean.

This module holds ONLY the pure decision: given the just-finished wave's infra
count and the current (already-adapted) settings, what should the next wave use?
Keeping it side-effect-free makes the backoff/recovery policy directly unit
testable; the orchestrator owns measuring the count and applying the result.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class WaveTuning:
    """The two knobs the orchestrator adapts between waves.

    ``max_workers`` is the concurrency cap; ``timeout_factor`` multiplies the
    configured gate/setup timeouts (1.0 == the configured values).
    """

    max_workers: int
    timeout_factor: float


def next_wave_tuning(
    infra_count: int,
    current: WaveTuning,
    *,
    base_workers: int,
    threshold: int = 1,
    timeout_factor: float = 2.0,
    max_timeout_factor: float = 4.0,
    min_workers: int = 1,
) -> WaveTuning:
    """Next wave's tuning from the last wave's infra count and current settings.

    - Infra count ABOVE ``threshold``: back off — halve workers (floor
      ``min_workers``) and grow the timeout factor by ``timeout_factor`` (ceiling
      ``max_timeout_factor``).
    - A CLEAN wave (count 0): recover gradually toward the configured values —
      double workers (ceiling ``base_workers``) and shrink the timeout factor by
      ``timeout_factor`` (floor 1.0).
    - Nonzero but AT/BELOW ``threshold``: hold steady, so a single self-healed
      hiccup neither panics nor prematurely recovers.

    Pure and idempotent at the bounds: at full concurrency a clean wave stays at
    full concurrency; at the floor a bad wave stays at the floor.
    """
    base_workers = max(1, int(base_workers))
    floor = max(1, int(min_workers))
    ceil = max(1.0, float(max_timeout_factor))
    step = max(1.0, float(timeout_factor))
    cur_workers = max(floor, int(current.max_workers))
    cur_factor = min(ceil, max(1.0, float(current.timeout_factor)))

    if infra_count > threshold:
        return WaveTuning(max(floor, cur_workers // 2), min(ceil, cur_factor * step))
    if infra_count == 0:
        return WaveTuning(
            min(base_workers, cur_workers * 2), max(1.0, cur_factor / step)
        )
    return WaveTuning(cur_workers, cur_factor)
