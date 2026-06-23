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
from pathlib import Path
from typing import Optional

from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)

# Cap on stored entries. One file per entry would otherwise grow without bound
# across many builds; past the cap the oldest entries (by mtime) are evicted.
# A re-applied hit only ever saves a gate run, so dropping cold entries is safe.
DEFAULT_MAX_ENTRIES = 2000


class LLMCache:
    """One JSON file per cache entry under a directory, bounded by mtime LRU."""

    def __init__(self, dir_path: Path, max_entries: int = DEFAULT_MAX_ENTRIES):
        self.dir = Path(dir_path)
        self.max_entries = max_entries

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
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Ignoring unreadable cache entry {path.name}: {e}")
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
        from my_project_orchestrator.utils.file_utils import ensure_artifact_dir

        ensure_artifact_dir(self.dir)
        key = self._key(system_prompt, prompt)
        path = self._path(key)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {"output": output, "model": model, "created_at": timestamp, "key": key},
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp.replace(path)
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """Drop the oldest entries (by mtime) once the cap is exceeded.

        Best-effort: any filesystem error degrades to leaving the cache as-is
        rather than raising into the build, like the rest of this module.
        """
        if self.max_entries <= 0:
            return
        try:
            entries = [
                (e.stat().st_mtime, e.path)
                for e in os.scandir(self.dir)
                if e.name.endswith(".json") and e.is_file()
            ]
        except OSError:
            return
        excess = len(entries) - self.max_entries
        if excess <= 0:
            return
        entries.sort(key=lambda t: t[0])
        for _mtime, victim in entries[:excess]:
            try:
                os.remove(victim)
            except OSError:
                continue
