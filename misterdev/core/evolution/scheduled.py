"""Schedulable entrypoint for one live evolution pass, gated behind the benchmark.

`run_evolution(..., live=True)` already promotes a self-edit ONLY when it beats the
champion on the benchmark. What was missing is a component that decides WHEN to run
it: a callable an external scheduler (cron/CI/`/loop`) can invoke, guarded by an
exclusive lock so overlapping scheduled runs cannot corrupt the shared archive or
worktree. A busy lock is a clean no-op.
"""

import json
import os
from pathlib import Path
from typing import Callable, Optional

from misterdev.core.evolution.driver import run_evolution
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)

_CIRCUIT_BREAKER_MAX_FAILURES = 5  # open circuit after this many consecutive failures
_CIRCUIT_BREAKER_RESET_RUNS = 1  # one success resets the counter


def _cb_state_path(lock: Path) -> Path:
    return lock.with_name("scheduled_health.json")


def _cb_read(lock: Path) -> int:
    """Consecutive failure count from the circuit-breaker state file."""
    p = _cb_state_path(lock)
    try:
        return int(
            json.loads(p.read_text(encoding="utf-8")).get("consecutive_failures", 0)
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return 0


def _cb_write(lock: Path, consecutive: int) -> None:
    p = _cb_state_path(lock)
    try:
        p.write_text(
            json.dumps({"consecutive_failures": consecutive}), encoding="utf-8"
        )
    except OSError:
        pass


def run_scheduled_evolution(
    project,
    benchmark_dir: str,
    workdir: str,
    *,
    gate_commands: dict,
    lock_path: Optional[str] = None,
    _run: Callable = run_evolution,
    **kwargs,
):
    """Run ONE live, benchmark-gated evolution pass under an exclusive lock.

    Returns the :class:`EvolutionResult`, or ``None`` when another scheduled pass
    already holds the lock (overlap is skipped, never queued). ``_run`` is the
    evolution entrypoint, injectable for tests. Always ``live=True`` — a scheduled
    pass only ever applies a change that passes the benchmark gate.
    """
    lock = Path(
        lock_path
        or (Path(project.path) / ".orchestrator" / "evolution" / "scheduled.lock")
    )
    lock.parent.mkdir(parents=True, exist_ok=True)

    consecutive = _cb_read(lock)
    if consecutive >= _CIRCUIT_BREAKER_MAX_FAILURES:
        logger.warning(
            "Scheduled evolution: circuit breaker open after %d consecutive failures; "
            "skipping this trigger. Fix the benchmark infrastructure to reset.",
            consecutive,
        )
        return None

    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
    except FileExistsError:
        # Check whether the lock holder is still alive; clean up if stale.
        stale = False
        try:
            pid = int(lock.read_text().strip())
            os.kill(pid, 0)
        except (OSError, ValueError):
            stale = True
        if stale:
            try:
                lock.unlink()
            except OSError:
                pass
            try:
                fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
            except OSError:
                return None
        else:
            logger.warning(
                "Scheduled evolution: another pass holds %s; skipping this trigger.",
                lock,
            )
            return None
    try:
        result = _run(
            project,
            benchmark_dir,
            workdir,
            gate_commands=gate_commands,
            live=True,
            **kwargs,
        )
        _cb_write(lock, 0)
        return result
    except Exception:
        _cb_write(lock, consecutive + 1)
        raise
    finally:
        os.close(fd)
        try:
            lock.unlink()
        except OSError:
            pass
