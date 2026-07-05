"""Disk-backed per-file symbol cache, keyed by content hash + lang + format."""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

from misterdev.utils.file_utils import atomic_write_json

from ._log import logger
from .nodes import SymbolNode, _symbol_to_dict, _symbol_from_dict

# Bump when the parse output shape or any _traverse_* logic changes, so a stale
# on-disk cache from an older grammar/format is discarded rather than served.
_CACHE_FORMAT_VERSION = 1


class _TopographyCache:
    """Disk-backed per-file symbol cache, keyed by content hash + lang + format.

    Maps a project-relative path to ``{"key": <hash>, "symbols": [...]}`` where
    the key folds the file's content sha256, its language, and the cache-format
    version. An entry is only reused when the recomputed key matches, so an edit
    (content change), a language change, or a format bump all invalidate it; mtime
    is never consulted. Every read/write is best-effort: a corrupt or unreadable
    cache degrades to an empty in-memory map and a full parse, never a raise and
    never stale symbols.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def make_key(content_bytes: bytes, lang: str) -> str:
        h = hashlib.sha256()
        h.update(f"{_CACHE_FORMAT_VERSION}\x00{lang}\x00".encode("utf-8"))
        h.update(content_bytes)
        return h.hexdigest()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as e:
            logger.debug(f"Topography cache unreadable, starting fresh: {e}")
            self.entries = {}
            return
        if (
            not isinstance(data, dict)
            or data.get("format") != _CACHE_FORMAT_VERSION
            or not isinstance(data.get("files"), dict)
        ):
            logger.debug("Topography cache format mismatch; starting fresh")
            self.entries = {}
            return
        self.entries = data["files"]

    def get(self, rel_path: str, key: str) -> Optional[List["SymbolNode"]]:
        """Cached symbols for ``rel_path`` iff its key matches, else None."""
        entry = self.entries.get(rel_path)
        if not isinstance(entry, dict) or entry.get("key") != key:
            return None
        raw = entry.get("symbols")
        if not isinstance(raw, list):
            return None
        try:
            return [_symbol_from_dict(d) for d in raw]
        except (KeyError, TypeError) as e:
            logger.debug(f"Topography cache entry for {rel_path} malformed: {e}")
            return None

    def put(self, rel_path: str, key: str, symbols: List["SymbolNode"]) -> None:
        self.entries[rel_path] = {
            "key": key,
            "symbols": [_symbol_to_dict(s) for s in symbols],
        }

    def prune(self, live_paths: Set[str]) -> None:
        """Drop entries for files no longer present in the source tree."""
        for stale in [p for p in self.entries if p not in live_paths]:
            del self.entries[stale]

    def save(self) -> None:
        try:
            atomic_write_json(
                self.path, {"format": _CACHE_FORMAT_VERSION, "files": self.entries}
            )
        except OSError as e:
            logger.debug(f"Topography cache write failed (non-fatal): {e}")
