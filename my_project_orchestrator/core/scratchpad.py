"""Scratchpad learning system ported from /build skill Phase 4.

Accumulates discoveries within a build session so later tasks can
benefit from earlier learnings. Categories match /build:
  env_quirk, pattern, dependency, convention, workaround, pitfall
"""

import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScratchpadEntry:
    category: str  # env_quirk, pattern, dependency, convention, workaround, pitfall
    discovery: str
    task_id: str
    files: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def matches(self, query_files: list[str] = None, query_tags: list[str] = None) -> bool:
        """Check if this entry is relevant to the given files or tags."""
        if query_files:
            for qf in query_files:
                for ef in self.files:
                    if qf in ef or ef in qf:
                        return True
        if query_tags:
            for qt in query_tags:
                if qt in self.tags or qt in self.category:
                    return True
        return False


class Scratchpad:
    """In-session learning accumulator.

    Before each task, query for relevant entries. After each task,
    record discoveries. This mirrors /build's scratchpad behavior
    but as a proper Python data structure.
    """

    def __init__(self):
        self._entries: list[ScratchpadEntry] = []
        self._lock = threading.Lock()

    def record(
        self,
        category: str,
        discovery: str,
        task_id: str,
        files: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
    ) -> ScratchpadEntry:
        """Record a discovery from a task execution."""
        entry = ScratchpadEntry(
            category=category,
            discovery=discovery,
            task_id=task_id,
            files=files or [],
            tags=tags or [],
        )
        with self._lock:
            self._entries.append(entry)
        return entry

    def query(
        self,
        files: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        category: Optional[str] = None,
    ) -> list[ScratchpadEntry]:
        """Find relevant entries for a task about to execute."""
        results = []
        with self._lock:
            snapshot = list(self._entries)
        for entry in snapshot:
            if category and entry.category != category:
                continue
            if files or tags:
                if entry.matches(query_files=files, query_tags=tags):
                    results.append(entry)
            elif not files and not tags:
                # No filter, return all (optionally filtered by category)
                results.append(entry)
        return results

    def format_context(
        self,
        files: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        max_entries: int = 20,
    ) -> str:
        """Format relevant entries as context string for LLM prompts."""
        entries = self.query(files=files, tags=tags)
        if not entries:
            return ""
        entries = entries[:max_entries]
        lines = ["## Scratchpad (learnings from previous tasks)"]
        for e in entries:
            lines.append(f"- [{e.category}] {e.discovery} (from {e.task_id})")
        return "\n".join(lines)

    @property
    def entries(self) -> list[ScratchpadEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)
