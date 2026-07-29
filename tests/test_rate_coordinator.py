"""Tests for the per-model rate-limit coordinator."""

import time

import misterdev.llm.client.rate_coordinator as _rc
from misterdev.llm.client.rate_coordinator import (
    clear,
    is_on_cooldown,
    record_cooldown,
    record_rate_limit,
    wait_if_needed,
)

import pytest


@pytest.fixture(autouse=True)
def _reset():
    clear()
    yield
    clear()


def _epoch(model: str) -> float:
    with _rc._lock:
        return _rc._retry_after.get(model, 0.0)


def test_wait_if_needed_no_op_when_no_backoff():
    t0 = time.time()
    wait_if_needed("some/model")
    assert time.time() - t0 < 0.1


def test_record_rate_limit_sets_epoch():
    t0 = time.time()
    record_rate_limit("m", 10.0)
    epoch = _epoch("m")
    assert epoch >= t0 + 9.9
    assert epoch <= t0 + 10.1


def test_record_rate_limit_extend_never_shorten():
    record_rate_limit("m", 100.0)
    epoch_after_large = _epoch("m")
    record_rate_limit("m", 5.0)  # smaller — must not move epoch earlier
    assert _epoch("m") == epoch_after_large


def test_record_rate_limit_larger_delay_extends():
    record_rate_limit("m", 10.0)
    epoch_small = _epoch("m")
    record_rate_limit("m", 200.0)  # larger — must push epoch later
    assert _epoch("m") > epoch_small


def test_cooldown_set_and_detected():
    assert not is_on_cooldown("m")
    record_cooldown("m", seconds=3600.0)
    assert is_on_cooldown("m")


def test_cooldown_expires():
    record_cooldown("m", seconds=-1.0)  # already in the past
    assert not is_on_cooldown("m")


def test_clear_all():
    record_cooldown("a", seconds=3600.0)
    record_cooldown("b", seconds=3600.0)
    clear()
    assert not is_on_cooldown("a")
    assert not is_on_cooldown("b")
    with _rc._lock:
        assert not _rc._retry_after


def test_clear_single_model():
    record_cooldown("a", seconds=3600.0)
    record_cooldown("b", seconds=3600.0)
    record_rate_limit("a", 100.0)
    clear("a")
    assert not is_on_cooldown("a")
    assert _epoch("a") == 0.0
    assert is_on_cooldown("b")  # b untouched


def test_backoff_and_cooldown_are_independent():
    record_rate_limit("m", 100.0)
    assert not is_on_cooldown("m")
    record_cooldown("m", seconds=3600.0)
    assert is_on_cooldown("m")
    assert _epoch("m") > time.time()  # backoff still set
