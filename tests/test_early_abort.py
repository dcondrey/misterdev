"""Process-signal early-abort: stop a non-converging task before the cost cap."""

from misterdev.core.execution.early_abort import (
    AttemptSignal,
    ConvergenceMonitor,
    fingerprint,
)


def test_fingerprint_ignores_incidental_detail():
    a = fingerprint("File /a/b.py line 42: NameError 'foo' at 0xdeadbeef")
    b = fingerprint("File /c/d.py line 99: NameError 'foo' at 0xcafef00d")
    assert a == b  # same error, different paths/lines/addresses -> same fingerprint


def test_stuck_fires_after_repeats():
    m = ConvergenceMonitor(stuck_repeats=3, no_progress_window=0)
    for _ in range(2):
        m.update(AttemptSignal(5, "TypeError: bad arg at line 3"))
        assert not m.should_abort()[0]  # not yet 3 in a row
    m.update(AttemptSignal(5, "TypeError: bad arg at line 7"))
    abort, reason = m.should_abort()
    assert abort and "stuck" in reason


def test_stuck_resets_when_error_changes():
    m = ConvergenceMonitor(stuck_repeats=3, no_progress_window=0)
    m.update(AttemptSignal(1, "error A"))
    m.update(AttemptSignal(1, "error A"))
    m.update(AttemptSignal(1, "error B"))  # different -> streak broken
    assert not m.should_abort()[0]


def test_no_progress_fires_when_count_not_shrinking():
    m = ConvergenceMonitor(stuck_repeats=0, no_progress_window=3)
    for c in (4, 4, 5):  # never strictly decreases
        m.update(AttemptSignal(c, "some error"))
    abort, reason = m.should_abort()
    assert abort and "no progress" in reason


def test_no_progress_does_not_fire_when_shrinking():
    m = ConvergenceMonitor(stuck_repeats=0, no_progress_window=3)
    for c in (6, 4, 2):  # strictly decreasing -> making progress
        m.update(AttemptSignal(c, "e"))
    assert not m.should_abort()[0]


def test_needs_evidence_before_aborting():
    m = ConvergenceMonitor(stuck_repeats=3, no_progress_window=3)
    m.update(AttemptSignal(9, "boom"))
    assert not m.should_abort()[0]  # a single attempt never aborts


def test_triggers_can_be_disabled():
    m = ConvergenceMonitor(stuck_repeats=0, no_progress_window=0)
    for _ in range(10):
        m.update(AttemptSignal(5, "same error"))
    assert not m.should_abort()[0]  # both triggers off -> never aborts
