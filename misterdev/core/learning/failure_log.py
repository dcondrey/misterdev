"""A durable, fingerprinted stream of real build failures.

The code evolver (:mod:`misterdev.core.evolution`) mines failures to aim a
self-edit at a real weakness — but until now it could only mine the synthetic
polyglot benchmark, because a finished build discarded its failure detail: the
saved report keeps only failed-task *ids* (``report.to_dict``), dropping the
error text, language, and category that failure attribution consumes. So
misterdev never learned about its own code from actual use.

This module closes that gap. Each finished build appends one record per failed
task to ``.orchestrator/failures.jsonl`` in the shape failure attribution already
duck-types on (``.name`` / ``.language`` / ``.resolved`` / ``.error``), so the
same targeting that runs over the benchmark runs over real failures unchanged.

Two properties make the stream a usable learning signal rather than a raw log:

* **Fingerprint recurrence** — each error is normalized (line numbers, hex
  addresses, temp paths, and bare digits stripped) to a stable fingerprint, so
  "wrong type at line 42" and "wrong type at line 99" collapse to one recurring
  failure. A failure that keeps recurring across runs is the highest-value
  target; a one-off is noise.
* **Recency decay** — a record's weight falls off with the number of runs since
  it was seen, so attribution chases what is breaking *now*, not what broke and
  was already fixed months ago.

Everything is best-effort: a missing or corrupt file loads as empty, a write
failure is swallowed. A learning stream must never be able to fail a build.
"""

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from misterdev.core.execution.error_classifier import classify_error
from misterdev.logging_setup import setup_logger
from misterdev.utils.file_utils import atomic_write

logger = setup_logger(__name__)

_MAX_RECORDS = (
    500  # bound the file; oldest rotate out (recency decay makes them worthless anyway)
)
_ERROR_CHARS = (
    1200  # keep enough of an error to classify + fingerprint, not a whole log
)
_HALF_LIFE_RUNS = 5.0  # runs after which a failure's recency weight halves

# File extension -> language, so a task's touched files infer the niche when the
# Task itself carries no language (attribution bins by language).
_EXT_LANG = {
    ".rs": "rust",
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".cs": "csharp",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
}

# Volatile substrings that make two instances of the SAME failure look distinct;
# stripped before fingerprinting so recurrence actually collapses. The order and
# anchoring matter: DISCRIMINATING digits (error codes like E0308, exit codes)
# must survive, only truly volatile numbers (line/column, offsets, addresses)
# are normalized.
_HEX = re.compile(r"0x[0-9a-fA-F]+")
# A long hex run is an address/hash ONLY when it contains a hex letter; a pure
# decimal of the same length is a count/size and must normalize like any number,
# so the same error isn't fingerprinted differently by the magnitude of a value.
_ADDR = re.compile(r"\b(?=[0-9]*[a-fA-F])[0-9a-fA-F]{8,}\b")
# Only STANDALONE digit runs (not adjacent to letters/digits), so a line number
# collapses but an error code's digits (E0308, E0499) are preserved and stay
# distinct — those are the most discriminating part of a compiler error.
_NUM = re.compile(r"(?<![0-9A-Za-z])\d+(?![0-9A-Za-z])")
_TMP = re.compile(r"(/tmp|/var/folders|[A-Za-z]:\\Temp)\S*", re.IGNORECASE)
_WS = re.compile(r"\s+")


def language_of(files: List[str], default: str = "unknown") -> str:
    """Infer a language from a task's touched files (first recognized extension).

    Bins a real failure into the same language niche the benchmark uses. Returns
    ``default`` when no file carries a known extension.
    """
    for f in files or []:
        ext = Path(str(f)).suffix.lower()
        if ext in _EXT_LANG:
            return _EXT_LANG[ext]
    return default


def fingerprint(error: str) -> str:
    """A stable short hash of an error with volatile detail normalized away.

    Two runs that hit the same failure at different line numbers / addresses /
    temp paths produce the same fingerprint, so recurrence can be counted. An
    empty error fingerprints to ``""`` (no signal, not a spurious match).
    """
    text = (error or "").strip().lower()
    if not text:
        return ""
    text = _TMP.sub("<path>", text)
    text = _HEX.sub("<hex>", text)
    text = _ADDR.sub("<addr>", text)
    text = _NUM.sub("<n>", text)
    text = _WS.sub(" ", text).strip()
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


@dataclass
class FailureRecord:
    """One real build failure, in the shape failure attribution consumes.

    ``resolved`` is always ``False`` (this is a failure log); it exists so the
    record duck-types as a benchmark result for :func:`attribution.attribute`.
    ``run`` is a monotonic counter used only for recency decay.
    """

    name: str  # the failed task's id
    language: str
    error: str
    category: str  # classified error category (e.g. "rust/wrong_type" -> "wrong_type")
    task_category: str = ""  # the task's own category (fix/feature/...) for context
    files: List[str] = field(default_factory=list)
    fp: str = ""  # error fingerprint (recurrence key)
    run: int = 0
    resolved: bool = False

    @classmethod
    def from_task(cls, task, run: int) -> Optional["FailureRecord"]:
        """Build a record from a failed :class:`~misterdev.core.models.Task`.

        Reuses the report's failure-reason extraction so the error text matches
        what the human report shows. Returns ``None`` when no error detail is
        recoverable — a failure with no signal teaches nothing and is not logged.
        """
        from misterdev.core.reporting.report import _failure_reason

        error = _failure_reason(task)
        if not error or error in ("failed", "pending", "in_progress"):
            # No recoverable detail: skip rather than log a contentless record.
            history = getattr(task, "execution_history", None) or []
            raw = ""
            if history:
                last = history[-1]
                raw = (
                    getattr(last, "logs", "") or getattr(last, "message", "")
                ).strip()
            if not raw:
                return None
            error = raw[:_ERROR_CHARS]
        error = error[:_ERROR_CHARS]
        files = list(getattr(task, "files_to_modify", []) or []) + list(
            getattr(task, "files_to_create", []) or []
        )
        return cls(
            name=str(getattr(task, "id", "") or "unknown"),
            language=language_of(files),
            error=error,
            category=classify_error(error),
            task_category=str(getattr(task, "category", "") or ""),
            files=files[:20],
            fp=fingerprint(error),
            run=run,
        )

    @property
    def output(self) -> str:
        """Alias so attribution's ``.output``-first read finds the error text."""
        return self.error

    def recency_weight(
        self, current_run: int, half_life: float = _HALF_LIFE_RUNS
    ) -> float:
        """Exponential recency weight in (0, 1]: 1.0 this run, halving every
        ``half_life`` runs. Keeps attribution aimed at what is breaking now."""
        age = max(0, current_run - self.run)
        return 0.5 ** (age / half_life) if half_life > 0 else 1.0


class FailureLog:
    """Append-only, bounded JSONL of real build failures at a project."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _load_raw(self) -> List[dict]:
        if not self.path.exists():
            return []
        rows: List[dict] = []
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue  # one bad line must not lose the whole stream
                if isinstance(obj, dict) and obj.get("name"):
                    rows.append(obj)
        except OSError as e:
            logger.warning(f"Failure log unreadable, treating as empty: {e}")
            return []
        return rows

    def load(self) -> List[FailureRecord]:
        """All records, oldest first. Degrades to empty on any read failure."""
        out: List[FailureRecord] = []
        for obj in self._load_raw():
            out.append(
                FailureRecord(
                    name=str(obj.get("name", "")),
                    language=str(obj.get("language", "unknown")),
                    error=str(obj.get("error", "")),
                    category=str(obj.get("category", "")),
                    task_category=str(obj.get("task_category", "")),
                    files=list(obj.get("files", []) or []),
                    fp=str(obj.get("fp", "")),
                    run=int(obj.get("run", 0)),
                )
            )
        return out

    def next_run(self) -> int:
        """The run number a new batch should carry (max existing + 1)."""
        rows = self._load_raw()
        return (max((int(r.get("run", 0)) for r in rows), default=0)) + 1

    def record_failures(self, failed_tasks) -> int:
        """Append one record per failed task under a fresh run number.

        Returns the number of records written. Best-effort: any error is logged
        and swallowed so a build is never failed by its own bookkeeping.
        """
        try:
            lock_path = self.path.with_suffix(".lock")
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                try:
                    import fcntl

                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
                except (ImportError, OSError):
                    pass
                # One read: derive the next run number and the existing rows together
                # (avoids a second full parse via next_run()).
                existing = self._load_raw()
                run = max((int(r.get("run", 0)) for r in existing), default=0) + 1
                records = [
                    r
                    for r in (FailureRecord.from_task(t, run) for t in failed_tasks)
                    if r
                ]
                if not records:
                    return 0
                # Bound the file: keep the most recent _MAX_RECORDS (tail), since
                # recency decay makes older records worthless to attribution anyway.
                combined = (existing + [asdict_record(r) for r in records])[
                    -_MAX_RECORDS:
                ]
                atomic_write(
                    self.path,
                    "\n".join(json.dumps(row, ensure_ascii=False) for row in combined)
                    + "\n",
                )
                logger.info(
                    f"Failure log: recorded {len(records)} failure(s) (run {run})."
                )
                return len(records)
            finally:
                os.close(lock_fd)
        except (OSError, ValueError) as e:
            logger.warning(f"Failure log write failed (non-fatal): {e}")
            return 0

    def recurrence(self) -> Dict[str, int]:
        """Count of records per fingerprint — how often each distinct failure has
        recurred across all logged runs (empty fingerprints excluded)."""
        counts: Dict[str, int] = {}
        for row in self._load_raw():
            fp = str(row.get("fp", ""))
            if fp:
                counts[fp] = counts.get(fp, 0) + 1
        return counts


def asdict_record(record: FailureRecord) -> dict:
    """A JSON-serializable dict of a record (drops the computed ``resolved``)."""
    data = asdict(record)
    data.pop("resolved", None)
    return data
