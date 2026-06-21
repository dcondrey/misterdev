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

from my_project_orchestrator.logging_setup import setup_logger
from my_project_orchestrator.utils.file_utils import atomic_write

logger = setup_logger(__name__)


def compute_task_hash(task, project_path: Path) -> str:
    """Content hash of a task's inputs: its devplan file + target file mtimes.

    Used to detect when a previously-completed task needs re-running because
    its spec or the files it touches changed since it last succeeded.
    """
    h = hashlib.sha256()
    source_ref = getattr(task, "source_ref", None)
    if source_ref:
        try:
            h.update(Path(source_ref).read_bytes())
        except OSError:
            pass
    # Fold in the task's own spec so that LLM-decomposed tasks (which have no
    # source_ref and reuse generic ids like T-001 across separate builds) get a
    # hash that reflects their actual content. Without this, a fresh plan's
    # T-001 collides with a prior build's completed T-001 and is wrongly skipped.
    for attr in ("title", "description"):
        h.update(str(getattr(task, attr, "")).encode())
    for f in sorted(getattr(task, "files_to_create", [])):
        h.update(f"create:{f}".encode())
    for f in sorted(getattr(task, "files_to_modify", [])):
        h.update(f"modify:{f}".encode())
        fp = Path(project_path) / f
        if fp.exists():
            h.update(f"{f}:{fp.stat().st_mtime_ns}".encode())
    return h.hexdigest()[:16]


class ProgressTracker:
    """Persists build progress so a crash at task 18 resumes from 18, not 1."""

    def __init__(self, project_path: Path):
        self._file = project_path / ".orchestrator" / "progress.json"
        self._file.parent.mkdir(parents=True, exist_ok=True)
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
