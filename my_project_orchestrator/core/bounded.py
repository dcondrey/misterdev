"""Shared bounded best-effort runner for the optional gates.

Several optional gates — goal check, adversarial critic, runtime/web/vision/
mutation smoke, LSP diagnostics, and the MCP substrate — must be strictly
non-blocking: a slow or hung model, server, or subprocess can NEVER stall a
build. They all shared one shape: run the work in a daemon thread, join with a
hard timeout, and on timeout abandon the thread (it dies with the process) and
return a SKIP sentinel. This centralizes that shape so the call sites don't each
re-implement it.
"""

import threading
from typing import Callable, TypeVar

from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)

T = TypeVar("T")


def run_bounded(
    fn: Callable[[], T], timeout: float, default: T, what: str = "operation"
) -> T:
    """Run ``fn()`` in a daemon thread, returning its result or ``default``.

    Returns ``default`` when ``fn`` does not finish within ``timeout`` seconds
    (the thread is abandoned and dies with the process) or when it raises (logged
    at debug — a backstop, since ``fn`` is expected to handle its own expected
    failures and return an appropriate value itself). ``what`` names the
    operation for log context. The caller is guaranteed control back within
    ``timeout`` seconds and is never handed an exception.
    """
    box = {"result": default}

    def _worker() -> None:
        try:
            box["result"] = fn()
        except Exception as e:  # backstop: fn should catch its own failures
            logger.debug(f"{what} failed in bounded runner: {e}")
            box["result"] = default

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        logger.warning(f"{what} timed out after {timeout}s; skipping (never blocks).")
        return default

    return box["result"]
