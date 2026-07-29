"""T5.2 — a scheduled evolution pass runs live behind the benchmark, lock-guarded.

`run_scheduled_evolution` is the schedulable entrypoint: it runs ONE live evolution
pass (which only promotes a self-edit that passes the benchmark) under an exclusive
lock, so overlapping scheduled runs cannot corrupt the shared archive. A busy lock is
a clean no-op. The benchmark-gating itself lives in run_evolution (live path) and is
injected here as a fake.
"""

import os
from types import SimpleNamespace

from misterdev.core.evolution.scheduled import run_scheduled_evolution


def test_runs_one_live_benchmark_gated_pass(tmp_path):
    calls = {}

    def fake_run(project, bdir, wdir, *, gate_commands, live=False, **kw):
        calls["live"] = live
        calls["gate"] = gate_commands
        return "RESULT"

    out = run_scheduled_evolution(
        SimpleNamespace(path=tmp_path),
        "bench",
        "work",
        gate_commands={"test_command": "pytest"},
        _run=fake_run,
    )
    assert out == "RESULT"
    assert calls["live"] is True  # scheduled pass is ALWAYS live+benchmark-gated
    assert calls["gate"] == {"test_command": "pytest"}
    # The lock is released after the pass completes.
    assert not (tmp_path / ".orchestrator" / "evolution" / "scheduled.lock").exists()


def test_busy_lock_is_skipped(tmp_path):
    lock = tmp_path / ".orchestrator" / "evolution" / "scheduled.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(os.getpid()))
    called = {"n": 0}

    def fake_run(*a, **k):
        called["n"] += 1
        return "X"

    out = run_scheduled_evolution(
        SimpleNamespace(path=tmp_path), "b", "w", gate_commands={}, _run=fake_run
    )
    assert out is None  # overlap is skipped, not queued
    assert called["n"] == 0


def test_lock_released_even_when_pass_raises(tmp_path):
    def boom(*a, **k):
        raise RuntimeError("pass failed")

    try:
        run_scheduled_evolution(
            SimpleNamespace(path=tmp_path), "b", "w", gate_commands={}, _run=boom
        )
    except RuntimeError:
        pass
    assert not (tmp_path / ".orchestrator" / "evolution" / "scheduled.lock").exists()
