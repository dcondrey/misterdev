"""In-process registry for long-running build/run jobs.

MCP tools are synchronous: a ``build`` call blocks the client (and the server's
event loop) until the whole autonomous run finishes — minutes for a real build,
long enough to trip client timeouts, with no way to monitor or stop it. This
registry runs a build/run in a background daemon thread and hands back a
``run_id`` immediately, so a client can poll ``status`` and request ``stop``.

Deliberately NOT a port of the donor's job machinery, whose concurrency was the
source of its critical bugs. Two guards address those failure modes directly:

- **One job per project** (donor C-010/CLU-003: unguarded concurrent runs on the
  same directory corrupted files/state). ``start`` refuses to launch a second
  job for a path that already has one running.
- **No silent thread death** (donor C-007: double-resolve / unhandled rejection).
  The runner captures every exception and records it on the job, so a crashed
  build becomes an observable ``failed`` status, never a lost thread.

Stop is cooperative and reuses misterdev's existing graceful kill-switch rather
than interrupting the task loop: the ``stop_hook`` lowers the run's budget so the
next model call raises ``BudgetExceededError``, which the orchestrator already
catches and degrades to a partial report. In-flight work finishes; no new work
starts.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _same_path(a: str, b: str) -> bool:
    """True when two paths point at the same directory (resolved)."""
    try:
        return Path(a).expanduser().resolve() == Path(b).expanduser().resolve()
    except OSError:
        return a == b


@dataclass
class Job:
    """A single background build/run and its observable state."""

    run_id: str
    kind: str  # "build" | "run"
    project_path: str
    status: str = "running"  # running | succeeded | failed | stopped
    result: Optional[str] = None
    error: Optional[str] = None
    started_at: str = field(default_factory=_now)
    ended_at: Optional[str] = None
    stop_requested: bool = False
    _stop_hook: Optional[Callable[[], None]] = None
    _thread: Optional[threading.Thread] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "kind": self.kind,
            "project_path": self.project_path,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "stop_requested": self.stop_requested,
        }


class JobRegistry:
    """Thread-safe registry of background jobs, keyed by ``run_id``."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(
        self,
        kind: str,
        project_path: str,
        target: Callable[[], str],
        stop_hook: Optional[Callable[[], None]] = None,
    ) -> str:
        """Launch ``target`` in a daemon thread and return its ``run_id``.

        ``target`` runs the (synchronous) build/run and returns a report string.
        Raises ``RuntimeError`` if a job is already running for ``project_path``
        — one writer per project, so concurrent runs cannot corrupt its tree.
        """
        with self._lock:
            for j in self._jobs.values():
                if j.status == "running" and _same_path(j.project_path, project_path):
                    raise RuntimeError(
                        f"a {j.kind} job (run_id {j.run_id}) is already running for "
                        f"{project_path}; stop it before starting another"
                    )
            run_id = uuid.uuid4().hex[:12]
            job = Job(
                run_id=run_id,
                kind=kind,
                project_path=str(project_path),
                _stop_hook=stop_hook,
            )
            self._jobs[run_id] = job

        def _runner() -> None:
            try:
                report = target()
                with self._lock:
                    job.status = "stopped" if job.stop_requested else "succeeded"
                    job.result = report
            except Exception as e:  # noqa: BLE001 - a crashed run must be observable, not lost
                logger.error(f"Job {run_id} ({kind}) raised: {e}")
                with self._lock:
                    if job.stop_requested:
                        job.status = "stopped"
                        job.result = f"Stopped by request; partial: {e}"
                    else:
                        job.status = "failed"
                        job.error = str(e)
            finally:
                with self._lock:
                    job.ended_at = _now()

        thread = threading.Thread(
            target=_runner, name=f"misterdev-job-{run_id}", daemon=True
        )
        job._thread = thread
        thread.start()
        return run_id

    def get(self, run_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(run_id)

    def status(self, run_id: str) -> Optional[Dict[str, Any]]:
        job = self.get(run_id)
        return job.to_dict() if job else None

    def stop(self, run_id: str) -> bool:
        """Request cooperative cancellation of a running job.

        Returns True if a running job was signalled, False if the id is unknown
        or the job already finished. Idempotent: stopping twice is harmless.
        """
        with self._lock:
            job = self._jobs.get(run_id)
            if job is None or job.status != "running":
                return False
            job.stop_requested = True
            hook = job._stop_hook
        if hook is not None:
            try:
                hook()
            except Exception as e:  # noqa: BLE001 - stop must never raise into the caller
                logger.warning(f"Stop hook for job {run_id} failed (non-fatal): {e}")
        return True

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [j.to_dict() for j in self._jobs.values()]


# Process-wide registry shared by the MCP server's async tools.
registry = JobRegistry()
