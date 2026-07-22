"""Schedulable entrypoint for one live evolution pass, gated behind the benchmark.

`run_evolution(..., live=True)` already promotes a self-edit ONLY when it beats the
champion on the benchmark. What was missing is a component that decides WHEN to run
it: a callable an external scheduler (cron/CI/`/loop`) can invoke, guarded by an
exclusive lock so overlapping scheduled runs cannot corrupt the shared archive or
worktree. A busy lock is a clean no-op.
"""

import os
from pathlib import Path
from typing import Callable, Optional

from misterdev.core.evolution.driver import run_evolution
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)


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
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        logger.warning(
            "Scheduled evolution: another pass holds %s; skipping this trigger.", lock
        )
        return None
    try:
        return _run(
            project,
            benchmark_dir,
            workdir,
            gate_commands=gate_commands,
            live=True,
            **kwargs,
        )
    finally:
        os.close(fd)
        try:
            lock.unlink()
        except OSError:
            pass
