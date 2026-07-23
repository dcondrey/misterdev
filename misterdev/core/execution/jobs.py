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
    # Task-level progress the running build reports (0/0 until it does).
    tasks_done: int = 0
    tasks_total: int = 0
    phase: str = ""
    message: str = ""
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
            "tasks_done": self.tasks_done,
            "tasks_total": self.tasks_total,
            "phase": self.phase,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Job":
        """Reconstruct a job from its persisted form (no live thread/hook).

        A job persisted as ``running`` belonged to a process that has since died,
        so its thread no longer exists: load it as ``interrupted`` rather than
        implying it is still making progress in this process."""
        status = d.get("status", "interrupted")
        if status == "running":
            status = "interrupted"
        return cls(
            run_id=str(d.get("run_id", "")),
            kind=str(d.get("kind", "")),
            project_path=str(d.get("project_path", "")),
            status=status,
            result=d.get("result"),
            error=d.get("error"),
            started_at=str(d.get("started_at", _now())),
            ended_at=d.get("ended_at"),
            stop_requested=bool(d.get("stop_requested", False)),
            tasks_done=int(d.get("tasks_done", 0) or 0),
            tasks_total=int(d.get("tasks_total", 0) or 0),
            phase=str(d.get("phase", "")),
            message=str(d.get("message", "")),
        )


# Cap on retained FINISHED jobs. A long-lived MCP server runs many builds over
# its lifetime; without eviction every job — and the orchestrator/client/report
# its stop-hook closure pins — would live forever (the donor's CLU-004 leak).
# Running jobs are never evicted.
_DEFAULT_MAX_FINISHED = 50


class JobRegistry:
    """Thread-safe registry of background jobs, keyed by ``run_id``."""

    def __init__(
        self,
        max_finished: int = _DEFAULT_MAX_FINISHED,
        store_path: Optional[str] = None,
    ) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._max_finished = max(1, max_finished)
        # Optional durable store so run_ids/results survive an MCP server restart.
        self._store = Path(store_path).expanduser() if store_path else None
        if self._store is not None:
            self._load()

    def _load(self) -> None:
        """Load persisted jobs on startup. Best-effort: a missing/corrupt store
        yields an empty registry, never an error at import/boot."""
        try:
            if not self._store.exists():
                return
            import json

            data = json.loads(self._store.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            logger.warning(f"Job store unreadable ({e}); starting empty.")
            return
        for d in data if isinstance(data, list) else []:
            job = Job.from_dict(d)
            if job.run_id:
                self._jobs[job.run_id] = job

    def _save_locked(self) -> None:
        """Persist all jobs to the store (atomic). Caller holds ``self._lock``.
        Best-effort: a write failure is logged, never raised into a job's path."""
        if self._store is None:
            return
        try:
            import json
            import os

            self._store.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps([j.to_dict() for j in self._jobs.values()], indent=2)
            tmp = self._store.with_suffix(self._store.suffix + ".tmp")
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, self._store)
        except OSError as e:
            logger.warning(f"Could not persist job store (non-fatal): {e}")

    def _prune_finished_locked(self) -> None:
        """Evict the least-recently-FINISHED jobs beyond the retention cap.

        Caller must hold ``self._lock``. Ordered by ``ended_at`` (finish time),
        not insertion order: a long job that started first but finished last is
        the newest result and must be kept, not dropped.
        """
        finished = sorted(
            (
                (j.ended_at or "", rid)
                for rid, j in self._jobs.items()
                if j.status != "running"
            )
        )
        excess = len(finished) - self._max_finished
        for _, rid in finished[:excess] if excess > 0 else []:
            del self._jobs[rid]

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
            # Bound retention so finished jobs (and the resources their closures
            # pin) don't accumulate over the server's lifetime.
            self._prune_finished_locked()
            self._save_locked()

        # A target may optionally accept a progress reporter: a 1-arg target gets
        # ``report(**fields)`` bound to this run; a legacy 0-arg target is called
        # as before (fully backward compatible).
        try:
            import inspect

            wants_reporter = len(inspect.signature(target).parameters) >= 1
        except (TypeError, ValueError):
            wants_reporter = False

        def _report(**fields: Any) -> None:
            self.update_progress(run_id, **fields)

        def _runner() -> None:
            try:
                report = target(_report) if wants_reporter else target()
                with self._lock:
                    if job.stop_requested:
                        # build() catches the stop's budget-trip and returns its
                        # report normally; label it so the reader knows the run
                        # was cancelled, not that a real budget ran out.
                        job.status = "stopped"
                        job.result = "Stopped by request (partial).\n\n" + report
                    else:
                        job.status = "succeeded"
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
                    self._prune_finished_locked()
                    self._save_locked()

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

    def update_progress(
        self,
        run_id: str,
        *,
        done: Optional[int] = None,
        total: Optional[int] = None,
        phase: Optional[str] = None,
        message: Optional[str] = None,
    ) -> bool:
        """Record task-level progress for a running job (surfaced by ``status``).

        Returns False for an unknown id. Only the provided fields change, so a
        caller can update just the phase, or just the done count. Persisted so
        progress survives a restart too."""
        with self._lock:
            job = self._jobs.get(run_id)
            if job is None:
                return False
            if done is not None:
                job.tasks_done = int(done)
            if total is not None:
                job.tasks_total = int(total)
            if phase is not None:
                job.phase = str(phase)
            if message is not None:
                job.message = str(message)
            self._save_locked()
        return True

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
            self._save_locked()
        if hook is not None:
            try:
                hook()
            except Exception as e:  # noqa: BLE001 - stop must never raise into the caller
                logger.warning(f"Stop hook for job {run_id} failed (non-fatal): {e}")
        return True

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [j.to_dict() for j in self._jobs.values()]


def _default_job_store() -> str:
    """The durable job store path: ``$MISTERDEV_STATE_DIR/jobs.json`` or the
    conventional ``~/.misterdev/jobs.json`` (matching the project registry)."""
    import os

    base = os.environ.get("MISTERDEV_STATE_DIR") or str(Path.home() / ".misterdev")
    return str(Path(base) / "jobs.json")


# Process-wide registry shared by the MCP server's async tools. Persistent by
# default so run_ids/results survive a server restart; loading is best-effort and
# writes are deferred until a job actually runs.
registry = JobRegistry(store_path=_default_job_store())
