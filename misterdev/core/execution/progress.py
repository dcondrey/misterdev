"""Task progress persistence for crash recovery.

Saves completed and failed task IDs to disk after each task finishes.
On build start, checks for existing progress and resumes from where
the previous run left off.
"""

import hashlib
import json
import threading
from pathlib import Path
from typing import Dict, Optional, Set

from misterdev.logging_setup import setup_logger
from misterdev.utils.file_utils import (
    atomic_write,
    orchestrator_state_file,
)

logger = setup_logger(__name__)


def compute_task_hash(task, project_path: Path) -> str:
    """Content-addressed fingerprint of a task, used on a resume to decide whether
    a previously-completed task still counts as done or must re-run.

    Keyed on the task's SPEC (id + title/description/acceptance — so an intentional
    change to what the task should do re-runs it, and generic LLM-decomposed ids
    like T-001 don't collide across builds) plus the COMMITTED CONTENT of the files
    it owns. Content is stable across git checkouts, worktree merges, and mtime
    churn — unlike the stat mtime this used to fold in, which spuriously
    invalidated resumes — and the whole plan file is no longer hashed, so editing
    an unrelated task or the preamble does not re-run everything.
    """
    h = hashlib.sha256()
    h.update(str(getattr(task, "id", "")).encode())
    for attr in ("title", "description", "acceptance_criteria"):
        h.update(b"\x00")
        h.update(str(getattr(task, attr, "") or "").encode())
    files = sorted(
        set(getattr(task, "files_to_create", []) or [])
        | set(getattr(task, "files_to_modify", []) or [])
    )
    for f in files:
        h.update(f"\x00file:{f}".encode())
        fp = Path(project_path) / f
        try:
            if fp.is_file():
                with fp.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(65536), b""):
                        h.update(chunk)
            else:
                h.update(b"\x00ABSENT")
        except OSError:
            h.update(b"\x00UNREADABLE")
    return h.hexdigest()[:16]


class ProgressTracker:
    """Persists build progress so a crash at task 18 resumes from 18, not 1."""

    def __init__(self, project_path: Path):
        self._file = orchestrator_state_file(project_path, "progress.json")
        self.completed: Set[str] = set()
        self.failed: Set[str] = set()
        self.hashes: Dict[str, str] = {}
        self.splits: Dict[str, list] = {}
        self.conflict_counts: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if self._file.exists():
            try:
                data = json.loads(self._file.read_text(encoding="utf-8"))
                self.completed = set(data.get("completed", []))
                self.failed = set(data.get("failed", []))
                self.hashes = dict(data.get("hashes", {}))
                self.splits = dict(data.get("splits", {}))
                self.conflict_counts = dict(data.get("conflict_counts", {}))
            except (json.JSONDecodeError, OSError):
                self.completed = set()
                self.failed = set()
                self.hashes = {}
                return
            # One-shot reconciliation: completed is the single terminal state, so a
            # task recorded as BOTH completed and failed (a pre-fix poisoned ledger,
            # or a failure written after an earlier success) is healed by dropping
            # it from failed — completed wins. Persist immediately so an existing
            # poisoned progress.json self-heals on disk even if this run marks
            # nothing else.
            poisoned = self.failed & self.completed
            if poisoned:
                logger.info(
                    "Reconciling progress ledger: %d task(s) in both completed and "
                    "failed; keeping completed: %s",
                    len(poisoned),
                    ", ".join(sorted(poisoned)),
                )
                self.failed -= self.completed
                self._save()

    def _save(self):
        data = json.dumps(
            {
                "completed": sorted(self.completed),
                "failed": sorted(self.failed),
                "hashes": self.hashes,
                "splits": self.splits,
                "conflict_counts": self.conflict_counts,
            },
            indent=2,
        )
        atomic_write(self._file, data)

    def mark_completed(self, task_id: str, task_hash: Optional[str] = None):
        with self._lock:
            self.completed.add(task_id)
            self.failed.discard(task_id)
            if task_hash is not None:
                self.hashes[task_id] = task_hash
            self._save()

    def needs_rerun(self, task_id: str, current_hash: str) -> bool:
        """True if the task isn't completed, or its inputs changed since."""
        if task_id not in self.completed:
            for parent, parts in self.splits.items():
                if task_id in parts and parent in self.completed:
                    return False
            return True
        recorded = self.hashes.get(task_id)
        return recorded is None or recorded != current_hash

    def record_split(self, original_id: str, part_ids: list) -> None:
        with self._lock:
            self.splits[original_id] = list(part_ids)
            self._save()

    def record_conflict(self, task_id_a: str, task_id_b: str) -> None:
        key = ",".join(sorted([task_id_a, task_id_b]))
        with self._lock:
            self.conflict_counts[key] = self.conflict_counts.get(key, 0) + 1
            self._save()

    def conflict_count(self, task_id_a: str, task_id_b: str) -> int:
        key = ",".join(sorted([task_id_a, task_id_b]))
        return self.conflict_counts.get(key, 0)

    def mark_failed(self, task_id: str):
        with self._lock:
            # completed is the single terminal state and always wins: a task that
            # already succeeded must never be re-listed as failed (deferred
            # persists through this same path), or the ledger reports two terminal
            # states and a stale failed entry could shadow green work. A genuine
            # post-success regression is caught by needs_rerun via the content
            # hash, not by a failed entry.
            if task_id in self.completed:
                return
            self.failed.add(task_id)
            self._save()

    def is_done(self, task_id: str) -> bool:
        return task_id in self.completed

    def has_previous_run(self) -> bool:
        return bool(self.completed or self.failed)

    def reset(self):
        with self._lock:
            self.completed.clear()
            self.failed.clear()
            self.hashes.clear()
            self.splits.clear()
            self.conflict_counts.clear()
            if self._file.exists():
                self._file.unlink()

    def summary(self) -> str:
        return f"{len(self.completed)} completed, {len(self.failed)} failed"
