"""Append-only audit trail (structured JSONL).

Records one JSON line per significant event — command run + exit, edit applied,
gate result, tool call — to ``.orchestrator/audit.jsonl`` under the project
root. Each line carries an ISO-8601 timestamp, an event ``type``, and
event-specific details.

Append-only and crash-tolerant: :meth:`AuditTrail.record` NEVER raises into the
caller. An unwritable path, a full disk, or a serialization failure is logged at
debug and dropped — audit is observability, so a logging failure must never
break a build. This mirrors the never-hang/never-hard-fail discipline of
:mod:`misterdev.core.execution.container` and ``lsp``.

Defaults ON: it is pure-win observability with no behavioral effect (it only
appends to a gitignored file under ``.orchestrator/``) and degrades silently if
the path is unwritable, so enabling it by default cannot regress a build. A
caller that wants it off passes ``enabled=False`` (or a null trail).
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)

_AUDIT_DIRNAME = ".orchestrator"
_AUDIT_FILENAME = "audit.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditTrail:
    """Append-only JSONL writer for build events.

    One instance per build, rooted at the project path. Writes are guarded by a
    lock so concurrent gate threads cannot interleave partial lines. Construction
    never raises; the directory is created lazily on first write so an unwritable
    root degrades to a no-op instead of failing at setup.
    """

    def __init__(self, project_path: Path, enabled: bool = True):
        self.enabled = enabled
        self.path = Path(project_path) / _AUDIT_DIRNAME / _AUDIT_FILENAME
        self._lock = threading.Lock()

    def record(self, event_type: str, **details: Any) -> None:
        """Append one event line. Never raises.

        ``event_type`` is the event category (``command``, ``edit``, ``gate``,
        ``tool``, ``model``, ...); ``details`` are merged into the line. A
        timestamp and the type are always present. Any failure (bad path,
        non-serializable detail, I/O error) is logged at debug and swallowed.
        """
        if not self.enabled:
            return
        entry = {"ts": _now_iso(), "type": event_type}
        # Merge details, coercing anything non-serializable to a string so a
        # surprising value can never make the whole line unwritable.
        for key, value in details.items():
            try:
                json.dumps(value)
                entry[key] = value
            except (TypeError, ValueError):
                entry[key] = str(value)
        try:
            line = json.dumps(entry, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            logger.debug(f"Audit entry not serializable, dropped: {e}")
            return
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except OSError as e:
            logger.debug(f"Audit write failed (dropped): {e}")

    def record_command(self, command: str, ok: bool, cwd: Optional[str] = None) -> None:
        self.record("command", command=command, ok=ok, cwd=cwd)

    def record_edit(self, path: str, action: str = "apply") -> None:
        self.record("edit", path=path, action=action)
