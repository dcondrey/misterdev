"""Content-addressed memoization of gate-passing LLM outputs.

Only outputs that cleared the validation gates are stored, keyed by a hash of
the full prompt (system + user). Because the executor's prompt already embeds
the current file contents, interface contracts, and strategy, the key changes
whenever any input that affects the correct output changes — so a stale edit
auto-invalidates. A cache hit is still re-applied through the gates, so even a
collision or staleness can only cost a wasted gate run, never ship bad code.

The store is model-agnostic on purpose: a cheap (or free) model's successful
result is reusable regardless of which model would be picked next time.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)

# Cap on stored entries. One file per entry would otherwise grow without bound
# across many builds; past the cap the oldest entries (by mtime) are evicted.
# A re-applied hit only ever saves a gate run, so dropping cold entries is safe.
DEFAULT_MAX_ENTRIES = 2000


class LLMCache:
    """One JSON file per cache entry under a directory, bounded by mtime LRU."""

    def __init__(
        self,
        dir_path: Path,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_age_days: Optional[float] = 7.0,
    ):
        self.dir = Path(dir_path)
        self.max_entries = max_entries
        self._max_age_seconds = max_age_days * 86400.0 if max_age_days else None
        # Running count of stored entries, seeded by one real scan on the first
        # new put and re-synced whenever an eviction scan runs. Lets a put below
        # the cap skip the O(entries) scandir+stat that would otherwise run on
        # every write (O(entries^2) filesystem work across a build).
        self._count_estimate: Optional[int] = None
        self._sweep_stale_tmp()

    @staticmethod
    def _key(system_prompt: str, prompt: str) -> str:
        h = hashlib.sha256()
        h.update((system_prompt or "").encode("utf-8"))
        h.update(b"\x00")
        h.update((prompt or "").encode("utf-8"))
        return h.hexdigest()

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, system_prompt: str, prompt: str) -> Optional[str]:
        """Return a cached output for this prompt, or None on miss."""
        path = self._path(self._key(system_prompt, prompt))
        if not path.exists():
            return None
        if self._max_age_seconds is not None:
            try:
                if time.time() - path.stat().st_mtime > self._max_age_seconds:
                    logger.debug(
                        f"Cache entry {path.name} expired (mtime); discarding."
                    )
                    return None
            except OSError:
                pass
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Ignoring unreadable cache entry {path.name}: {e}")
            return None
        if self._max_age_seconds is not None:
            created = data.get("created_at", 0)
            if (
                isinstance(created, (int, float))
                and time.time() - created > self._max_age_seconds
            ):
                logger.debug(f"Cache entry {path.name} expired; discarding.")
                return None
        output = data.get("output")
        return output if isinstance(output, str) else None

    def put(
        self,
        system_prompt: str,
        prompt: str,
        output: str,
        model: str = "",
        timestamp: float = 0.0,
    ) -> None:
        """Store a gate-passing output (atomic write)."""
        from misterdev.utils.file_utils import atomic_write_json

        key = self._key(system_prompt, prompt)
        path = self._path(key)
        is_new = not path.exists()
        actual_ts = timestamp if timestamp > 0 else time.time()
        atomic_write_json(
            path,
            {"output": output, "model": model, "created_at": actual_ts, "key": key},
            indent=2,
        )
        if is_new:
            self._note_new_entry()

    def _note_new_entry(self) -> None:
        """Account for a newly-stored entry, scanning only when eviction is due.

        The full scandir+stat sweep runs only when the running count first needs
        seeding or crosses the cap — not on every put — so a long build's writes
        cost O(entries) filesystem work total rather than O(entries^2).
        """
        if self.max_entries <= 0:
            return
        if self._count_estimate is None:
            self._count_estimate = self._evict_if_needed()
            return
        self._count_estimate += 1
        if self._count_estimate > self.max_entries:
            self._count_estimate = self._evict_if_needed()

    def _evict_if_needed(self) -> int:
        """Drop the oldest entries (by mtime) once the cap is exceeded.

        Returns the entry count after any eviction, so the caller can re-sync its
        running estimate. Best-effort: any filesystem error degrades to leaving
        the cache as-is rather than raising into the build, like the rest of this
        module.
        """
        if self.max_entries <= 0:
            return 0
        try:
            entries = [
                (e.stat().st_mtime, e.path)
                for e in os.scandir(self.dir)
                if e.name.endswith(".json") and e.is_file()
            ]
        except OSError:
            return 0
        excess = len(entries) - self.max_entries
        if excess <= 0:
            return len(entries)
        entries.sort(key=lambda t: t[0])
        removed = 0
        for _mtime, victim in entries[:excess]:
            try:
                os.remove(victim)
                removed += 1
            except OSError:
                continue
        return len(entries) - removed

    def _sweep_stale_tmp(self) -> None:
        """Remove .json.tmp files left by crashed atomic writes (older than 5 min)."""
        if not self.dir.is_dir():
            return
        try:
            cutoff = time.time() - 300
            for entry in os.scandir(self.dir):
                if entry.name.endswith(".json.tmp"):
                    try:
                        if entry.stat().st_mtime < cutoff:
                            os.remove(entry.path)
                    except OSError:
                        pass
        except OSError:
            pass
