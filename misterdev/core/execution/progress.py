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
            h.update(fp.read_bytes() if fp.is_file() else b"\x00ABSENT")
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
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if self._file.exists():
            try:
                data = json.loads(self._file.read_text(encoding="utf-8"))
                self.completed = set(data.get("completed", []))
                self.failed = set(data.get("failed", []))
                self.hashes = dict(data.get("hashes", {}))
            except (json.JSONDecodeError, OSError):
                self.completed = set()
                self.failed = set()
                self.hashes = {}

    def _save(self):
        data = json.dumps(
            {
                "completed": sorted(self.completed),
                "failed": sorted(self.failed),
                "hashes": self.hashes,
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
            return True
        recorded = self.hashes.get(task_id)
        return recorded is None or recorded != current_hash

    def mark_failed(self, task_id: str):
        with self._lock:
            self.failed.add(task_id)
            self._save()

    def is_done(self, task_id: str) -> bool:
        return task_id in self.completed

    def has_previous_run(self) -> bool:
        return bool(self.completed or self.failed)

    def reset(self):
        self.completed.clear()
        self.failed.clear()
        self.hashes.clear()
        if self._file.exists():
            self._file.unlink()

    def summary(self) -> str:
        return f"{len(self.completed)} completed, {len(self.failed)} failed"
