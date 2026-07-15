"""Per-project environment memory that persists across runs.

Every run otherwise re-discovers the same durable facts about a project's
environment: which dependency-prime command actually works, how many parallel
workers the machine sustains before it thrashes (see ``adaptive.py``), which
toolchain-resolve probe is right, and any hard ordering constraint learned the
hard way. This file records those facts under ``.orchestrator/env_learnings.json``
so the NEXT run starts pre-tuned instead of relearning from scratch — the
cross-run self-improvement loop.

Schema (tiny and forward-compatible; every field optional):

    {
      "version": 1,
      "worktree_setup_command": "pnpm install --prefer-offline" | null,
      "worktree_healthcheck_command": "npx --no-install tsc --version" | null,
      "max_workers": 4 | null,            // a learned REDUCTION from backoff
      "ordering_constraints": ["dashboard build must precede server test"]
    }

Applying a learned value NEVER overrides an explicit ``project.yaml`` setting —
the user's config always wins; a learning only fills a gap the user left open.
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional

from misterdev.logging_setup import setup_logger
from misterdev.utils.file_utils import atomic_write, orchestrator_state_file

logger = setup_logger(__name__)

_VERSION = 1
_FILE = "env_learnings.json"

# (learning attribute, config section, config key). The only fields that pre-tune
# a runtime SETTING; ordering_constraints is persisted advisory data, not a knob.
_TUNABLES = (
    ("worktree_setup_command", "orchestrator", "worktree_setup_command"),
    ("worktree_healthcheck_command", "orchestrator", "worktree_healthcheck_command"),
    ("max_workers", "orchestrator", "max_workers"),
)


@dataclass
class EnvLearnings:
    """Durable, per-project environment facts learned across runs."""

    worktree_setup_command: Optional[str] = None
    worktree_healthcheck_command: Optional[str] = None
    max_workers: Optional[int] = None
    ordering_constraints: List[str] = field(default_factory=list)

    @classmethod
    def load(cls, project_path) -> "EnvLearnings":
        """Load the project's learnings, or an empty set. Never raises: a missing
        or corrupt file self-heals to empty so a bad ledger can't break a run."""
        path = orchestrator_state_file(project_path, _FILE)
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, ValueError) as e:
            logger.warning(f"Env-learnings unreadable ({e}); starting empty.")
            return cls()
        if not isinstance(data, dict):
            return cls()
        mw = data.get("max_workers")
        oc = data.get("ordering_constraints")
        return cls(
            worktree_setup_command=data.get("worktree_setup_command") or None,
            worktree_healthcheck_command=data.get("worktree_healthcheck_command")
            or None,
            max_workers=int(mw) if isinstance(mw, int) and mw > 0 else None,
            ordering_constraints=[str(c) for c in oc] if isinstance(oc, list) else [],
        )

    def save(self, project_path) -> None:
        """Persist atomically under ``.orchestrator/``. Best-effort."""
        path = orchestrator_state_file(project_path, _FILE)
        data = {
            "version": _VERSION,
            "worktree_setup_command": self.worktree_setup_command,
            "worktree_healthcheck_command": self.worktree_healthcheck_command,
            "max_workers": self.max_workers,
            "ordering_constraints": list(self.ordering_constraints),
        }
        atomic_write(path, json.dumps(data, indent=2))

    def apply_to_config(self, config: dict) -> List[str]:
        """Pre-tune ``config`` from learned values WITHOUT overriding explicit ones.

        For each tunable, a learned value is written into the config only when the
        user left that key unset (an explicit ``project.yaml`` value always wins).
        Returns a human-readable list of what was applied, for logging. Mutates the
        config dict in place (the same dict ``get_setting`` reads).
        """
        applied: List[str] = []
        for attr, section, key in _TUNABLES:
            value = getattr(self, attr)
            if value is None:
                continue
            sect = config.get(section)
            if isinstance(sect, dict) and key in sect:
                continue  # explicit config wins — never override it
            config.setdefault(section, {})[key] = value
            applied.append(f"{section}.{key}={value!r}")
        return applied
