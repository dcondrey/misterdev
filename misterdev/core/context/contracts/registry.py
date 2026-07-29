"""Contract data type and the cross-task contract registry."""

import json
import threading
from pathlib import Path
from typing import Dict, List

from misterdev.utils.file_utils import (
    atomic_write,
    orchestrator_state_file,
)

from ._log import logger
from .extraction import _extract_public_symbols


class Contract:
    """A public API contract extracted from a completed task."""

    def __init__(self, task_id: str, file_path: str, symbols: List[Dict[str, str]]):
        self.task_id = task_id
        self.file_path = file_path
        self.symbols = symbols  # [{name, kind, signature}]

    def format_for_prompt(self) -> str:
        lines = [f"### {self.file_path} (from {self.task_id})"]
        for sym in self.symbols:
            kind = sym.get("kind", "symbol")
            name = sym.get("name", "?")
            sig = sym.get("signature", "")
            lines.append(f"- {kind}: `{sig or name}`")
        return "\n".join(lines)


class ContractRegistry:
    """Manages interface contracts across tasks.

    After a task completes, call `extract_contracts()` to record what it exported.
    Before a task executes, call `get_contracts_for_task()` to get the interfaces
    it depends on.
    """

    def __init__(self, project_path: Path):
        self.contracts: Dict[str, List[Contract]] = {}  # task_id -> contracts
        self._file = orchestrator_state_file(project_path, "contracts.json")
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if self._file.exists():
            try:
                data = json.loads(self._file.read_text(encoding="utf-8"))
                for task_id, entries in data.items():
                    self.contracts[task_id] = [
                        Contract(task_id, e["file_path"], e["symbols"]) for e in entries
                    ]
            except (json.JSONDecodeError, OSError, KeyError) as e:
                logger.warning(
                    f"contracts.json unreadable, starting fresh: {self._file}: {e}"
                )
                self.contracts = {}

    def _save(self):
        data = {}
        for task_id, contracts in self.contracts.items():
            data[task_id] = [
                {"file_path": c.file_path, "symbols": c.symbols} for c in contracts
            ]
        atomic_write(self._file, json.dumps(data, indent=2))

    def extract_contracts(
        self,
        task_id: str,
        modified_files: List[str],
        project_path: Path,
        llm_client,
        language: str = "rust",
    ) -> List[Contract]:
        """Extract public API from files modified by a completed task."""
        contracts = []
        for file_path in modified_files:
            full_path = project_path / file_path
            try:
                full_path = full_path.resolve()
                full_path.relative_to(project_path.resolve())
            except (ValueError, OSError):
                logger.warning("Skipping out-of-bounds contract path: %s", file_path)
                continue
            if not full_path.exists():
                continue
            try:
                content = full_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            if len(content.strip()) == 0:
                continue

            symbols = _extract_public_symbols(content, language)
            if symbols:
                contracts.append(Contract(task_id, file_path, symbols))

        with self._lock:
            self.contracts[task_id] = contracts
            self._save()
        logger.info(
            f"Extracted {sum(len(c.symbols) for c in contracts)} contracts from {task_id}"
        )
        return contracts

    def get_contracts_for_task(self, dependency_ids: List[str]) -> str:
        """Format contracts from dependency tasks as prompt context."""
        if not dependency_ids:
            return ""

        relevant = []
        with self._lock:
            for dep_id in dependency_ids:
                if dep_id in self.contracts:
                    relevant.extend(self.contracts[dep_id])

        if not relevant:
            return ""

        lines = [
            "## Interface Contracts (from completed dependency tasks)",
            "Your code MUST use these exact signatures. Do not guess or assume different names.\n",
        ]
        for contract in relevant:
            lines.append(contract.format_for_prompt())
        return "\n".join(lines)

    def get_all_contracts_summary(self) -> str:
        """Summary for reporting."""
        total = sum(len(cs) for cs in self.contracts.values())
        return f"{len(self.contracts)} tasks, {total} total contracts"
