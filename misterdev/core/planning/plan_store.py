"""Persisted plan proposals with an explicit approval gate.

An MCP client can ask misterdev to *propose* work (analysis + recommendations)
without the codebase entering the client's context, review the proposals, then
*approve* a subset before any code is edited. That review step needs somewhere
to hold the proposals between calls; this module is it — a small JSON store
under the project's ``.orchestrator`` directory.

Pure and side-effect-scoped: it only reads/writes ``proposed_plan.json`` and
carries no LLM or execution logic, so the approval semantics are fully
unit-testable offline. Analysis (which produces proposals) and execution (which
acts on approvals) live in the orchestrator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from misterdev.logging_setup import setup_logger
from misterdev.utils.file_utils import atomic_write_json

logger = setup_logger(__name__)

_PLAN_FILE = "proposed_plan.json"


def _plan_path(project_path: str | Path) -> Path:
    return Path(project_path) / ".orchestrator" / _PLAN_FILE


def save_plan(
    project_path: str | Path, recommendations: List[Any]
) -> List[Dict[str, Any]]:
    """Persist ``recommendations`` as a fresh, unapproved proposed plan.

    Each recommendation may be a dict or an object with ``title`` /
    ``work_type`` / ``rationale`` attributes (e.g. advisor ``Recommendation``).
    Assigns stable ``P-001``-style ids and ``approved: false``, overwriting any
    prior plan. Returns the stored items.
    """
    items: List[Dict[str, Any]] = []
    for i, rec in enumerate(recommendations, 1):
        title = _field(rec, "title")
        if not title:
            continue
        items.append(
            {
                "id": f"P-{i:03d}",
                "title": title,
                "work_type": _field(rec, "work_type") or "complete",
                "rationale": _field(rec, "rationale") or "",
                "approved": False,
            }
        )
    _write(project_path, items)
    return items


def load_plan(project_path: str | Path) -> Optional[List[Dict[str, Any]]]:
    """Return the stored plan items, or None when no plan has been proposed."""
    path = _plan_path(project_path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.error(f"Could not read proposed plan at {path}: {e}")
        return None
    return data if isinstance(data, list) else None


def set_approval(
    project_path: str | Path,
    item_ids: Optional[List[str]] = None,
    approve_all: bool = False,
    reject_ids: Optional[List[str]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Update approval flags on the stored plan and persist.

    ``approve_all`` approves every item. Otherwise ``item_ids`` are approved and
    ``reject_ids`` are un-approved; an id in both is rejected (reject wins, the
    safer default). Returns the updated items, or None if no plan exists.
    """
    plan = load_plan(project_path)
    if plan is None:
        return None
    approve = set(item_ids or [])
    reject = set(reject_ids or [])
    for item in plan:
        if approve_all:
            item["approved"] = True
        if item["id"] in approve:
            item["approved"] = True
        if item["id"] in reject:
            item["approved"] = False
    _write(project_path, plan)
    return plan


def approved_items(project_path: str | Path) -> List[Dict[str, Any]]:
    """Return only the approved items (empty list when none / no plan)."""
    plan = load_plan(project_path) or []
    return [item for item in plan if item.get("approved")]


def _field(rec: Any, name: str) -> str:
    value = rec.get(name) if isinstance(rec, dict) else getattr(rec, name, None)
    return str(value).strip() if value else ""


def _write(project_path: str | Path, items: List[Dict[str, Any]]) -> None:
    path = _plan_path(project_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, items, indent=2)
