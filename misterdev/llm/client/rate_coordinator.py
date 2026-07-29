"""Per-model rate-limit coordination across parallel LLM workers.

Prevents thundering herd: when multiple workers hit a 429 on the same model they
all wait on the same shared backoff epoch rather than sleeping independently. A
non-retryable failure places the model on cooldown so the next routing decision
skips it without paying for another doomed call.
"""

import threading
import time

_lock = threading.Lock()
# model -> earliest epoch at which any worker may call it (429 backoff)
_retry_after: dict[str, float] = {}
# model -> earliest epoch at which routing may select it (non-retryable cooldown)
_cooldown: dict[str, float] = {}

_COOLDOWN_SECONDS = 60.0


def wait_if_needed(model: str) -> None:
    """Block the caller until the shared 429 backoff window for this model has elapsed."""
    with _lock:
        until = _retry_after.get(model, 0.0)
    remaining = until - time.time()
    if remaining > 0:
        time.sleep(remaining)


def record_rate_limit(model: str, delay: float) -> None:
    """Extend the shared backoff epoch when a worker receives a 429.

    Only extends, never shortens: if a later worker arrives with a larger
    delay it pushes the epoch out, not in.
    """
    with _lock:
        new_epoch = time.time() + delay
        if new_epoch > _retry_after.get(model, 0.0):
            _retry_after[model] = new_epoch


def record_cooldown(model: str, seconds: float = _COOLDOWN_SECONDS) -> None:
    """Mark model unavailable for routing after a non-retryable failure."""
    with _lock:
        _cooldown[model] = time.time() + seconds


def is_on_cooldown(model: str) -> bool:
    """True when model had a recent non-retryable failure and should be skipped."""
    with _lock:
        return time.time() < _cooldown.get(model, 0.0)


def clear(model: str | None = None) -> None:
    """Reset coordinator state. Used in tests to isolate state between cases."""
    with _lock:
        if model is None:
            _retry_after.clear()
            _cooldown.clear()
        else:
            _retry_after.pop(model, None)
            _cooldown.pop(model, None)
