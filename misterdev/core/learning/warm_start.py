"""Warm-start a new task from the approach of its nearest solved task.

A new task that resembles one already solved starts cold: misterdev re-derives an
approach it has produced before. This index records each solved task's shape
(goal, files touched, category) so a later, similar task can be seeded with "here
is how a task like this was solved before" instead of from scratch — the
smarter-*and*-faster lever, since a good prior cuts attempts-to-green.

Retrieval reuses the project's :class:`SemanticRanker`, so it is dense+lexical
when an embedder is available and lexical-only otherwise (never a hard
dependency). The index is append-only, deduplicated by normalized description
(the same recurring task doesn't accumulate duplicates), and bounded. Best-effort
throughout: any I/O failure degrades to "no warm-start", never a build failure.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from misterdev.core.economics.embeddings import EmbeddingCache, SemanticRanker
from misterdev.core.learning.failure_log import language_of
from misterdev.logging_setup import setup_logger
from misterdev.utils.file_utils import atomic_write

logger = setup_logger(__name__)

_MAX_SOLVED = 300  # bound the index; oldest solved tasks rotate out
_MAX_NEIGHBORS = 3  # solved tasks injected as priors for a new build
_TEXT_CHARS = 400


@dataclass
class SolvedTask:
    """One previously-solved task, enough to seed a similar future one."""

    task_id: str
    description: str
    language: str
    category: str = ""
    files: List[str] = field(default_factory=list)

    def text(self) -> str:
        """The text a query is matched against (goal + touched files)."""
        return f"{self.description} {' '.join(self.files)}".strip()


class SolvedTaskIndex:
    """Append-only, deduplicated, bounded index of solved tasks at a project."""

    def __init__(self, path: Path, embedder=None):
        self.path = Path(path)
        self.embedder = embedder

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
                    continue
                if isinstance(obj, dict) and obj.get("task_id"):
                    rows.append(obj)
        except OSError as e:
            logger.warning(f"Solved-task index unreadable, treating as empty: {e}")
            return []
        return rows

    def load(self) -> List[SolvedTask]:
        out: List[SolvedTask] = []
        for obj in self._load_raw():
            out.append(
                SolvedTask(
                    task_id=str(obj.get("task_id", "")),
                    description=str(obj.get("description", "")),
                    language=str(obj.get("language", "unknown")),
                    category=str(obj.get("category", "")),
                    files=list(obj.get("files", []) or []),
                )
            )
        return out

    def record(self, completed_tasks) -> int:
        """Append solved tasks, deduplicated by normalized description.

        The dedup key preserves digits (unlike the error fingerprint), so two
        genuinely different tasks that differ only by a number — "add migration
        0001" vs "0002" — stay distinct instead of evicting each other; only a
        re-solve of the same description is refreshed in place. Returns the count
        of genuinely new solved tasks. Best-effort: any failure is a no-op.
        """
        try:
            rows = self._load_raw()
            by_key: dict = {}
            for row in rows:
                key = str(row.get("key") or _dedup_key(row.get("description", "")))
                by_key[key] = row

            added = 0
            for task in completed_tasks:
                desc = _describe(task)
                if not desc:
                    continue
                key = _dedup_key(desc)
                files = list(getattr(task, "files_to_modify", []) or []) + list(
                    getattr(task, "files_to_create", []) or []
                )
                row = {
                    "task_id": str(getattr(task, "id", "") or "unknown"),
                    "description": desc[:_TEXT_CHARS],
                    "language": language_of(files),
                    "category": str(getattr(task, "category", "") or ""),
                    "files": files[:20],
                    "key": key,
                }
                if key not in by_key:
                    added += 1
                else:
                    # A re-solve moves the task to the freshest position so it
                    # isn't rotated out by the cap while newer tasks accumulate.
                    del by_key[key]
                by_key[key] = row

            ordered = list(by_key.values())[-_MAX_SOLVED:]
            if not ordered:
                return 0
            atomic_write(
                self.path,
                "\n".join(json.dumps(r, ensure_ascii=False) for r in ordered) + "\n",
            )
            return added
        except (OSError, ValueError) as e:
            logger.warning(f"Solved-task index write failed (non-fatal): {e}")
            return 0

    def nearest(self, query: str, k: int = _MAX_NEIGHBORS) -> List[SolvedTask]:
        """The ``k`` solved tasks most relevant to ``query`` (dense+lexical when an
        embedder is present, lexical otherwise). Empty query or empty index -> []."""
        solved = self.load()
        if not query or not solved:
            return []
        cache = (
            EmbeddingCache(
                self.path.with_name("solved_embeddings.json"),
                getattr(self.embedder, "model", "unknown"),
            )
            if self.embedder is not None
            else None
        )
        ranker = SemanticRanker(embedder=self.embedder, cache=cache)
        # Key by row position, NOT task_id: ids (e.g. "T-001") repeat across builds,
        # so keying by id would collapse the accumulated index to one row per slot.
        candidates = {str(i): s.text() for i, s in enumerate(solved)}
        return [solved[int(i)] for i in ranker.top_k(query, candidates, k)]

    def context(self, query: str, k: int = _MAX_NEIGHBORS) -> str:
        """A markdown block of nearest solved tasks for prompt injection, or ""."""
        neighbors = self.nearest(query, k)
        if not neighbors:
            return ""
        lines = ["## Similar Previously-Solved Tasks (warm-start priors)"]
        for s in neighbors:
            files = ", ".join(s.files[:5])
            lines.append(
                f"- [{s.language}] {s.description}"
                + (f" (touched: {files})" if files else "")
            )
        return "\n".join(lines)


_WS = re.compile(r"\s+")


def _dedup_key(description: str) -> str:
    """A stable dedup key for a task description that PRESERVES digits.

    Two tasks differing only by a number are distinct work (migration 0001 vs
    0002), so — unlike the error fingerprint, which strips digits to collapse
    line numbers — this normalizes only case and whitespace. Only a true re-solve
    of the same description collapses."""
    normalized = _WS.sub(" ", (description or "").strip().lower())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def _describe(task) -> str:
    """A task's matchable description: its title/goal, falling back to its id."""
    for attr in ("title", "description"):
        val = str(getattr(task, attr, "") or "").strip()
        if val:
            return val
    return str(getattr(task, "id", "") or "").strip()
