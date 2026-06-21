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
from pathlib import Path
from typing import Optional

from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)


class LLMCache:
    """One JSON file per cache entry under a directory."""

    def __init__(self, dir_path: Path):
        self.dir = Path(dir_path)

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
