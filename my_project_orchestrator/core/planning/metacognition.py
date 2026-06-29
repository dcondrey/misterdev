"""Metacognitive Trace Auditing and Project-Specific Learning.

Reviews build session history to generate permanent project-specific rules.
"""

import json
from pathlib import Path
from typing import List, Any

from my_project_orchestrator.logging_setup import setup_logger
from my_project_orchestrator.llm.client import BaseLLMClient
from my_project_orchestrator.llm.responses import extract_json_array

logger = setup_logger(__name__)


class SessionAuditor:
    """Audits task execution traces to extract 'Lessons Learned'."""

    def __init__(self, project_path: Path, llm_client: BaseLLMClient):
        self.project_path = project_path
        self.llm = llm_client
        self.lessons_file = project_path / ".orchestrator" / "lessons.json"

    def audit_session(
        self,
        completed_tasks: List[Any],
        failed_tasks: List[Any],
        scratchpad_content: str,
    ) -> str:
        """Runs the audit and returns a 'Project Logic Patch' string."""
        logger.info("Auditing build session for metacognitive learning...")

        prompt = f"""Review the following session traces for this project.
Identify any project-specific 'pitfalls', 'conventions', or 'requirements' that were discovered.

Successful Patterns:
{completed_tasks}

Failed Attempts / Stalls:
{failed_tasks}

Scratchpad Learnings:
{scratchpad_content}

Generate a list of 1-5 'Project Specific Rules' for future agents.
Example: 'Always run black before committing', 'The database connection must be closed manually in tests'.

Return a JSON array of strings. Return ONLY the JSON array.
"""
        try:
            response = self.llm.generate_code(
                prompt, "You are a senior project auditor."
            )
            new_rules = _extract_json_array(response)
            if new_rules:
                self._save_lessons(new_rules)
                return "\n".join(f"- {r}" for r in new_rules)
        except Exception as e:
            logger.error(f"Metacognitive audit failed: {e}")

        return "No new lessons learned."

    def _save_lessons(self, new_rules: List[str]):
        """Permanently appends rules to the project's lessons file."""
        self.lessons_file.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if self.lessons_file.exists():
            try:
                existing = json.loads(self.lessons_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = []

        # Coerce to strings before dedup: the LLM sometimes returns JSON objects
        # instead of plain rule strings, and set()-based dedup crashes on the
        # unhashable dicts (silently killing every audit). dict.fromkeys keeps
        # insertion order while removing duplicates.
        def _as_text(r):
            return r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)

        merged = [_as_text(r) for r in existing] + [_as_text(r) for r in new_rules]
        updated = list(dict.fromkeys(merged))
        self.lessons_file.write_text(json.dumps(updated, indent=2), encoding="utf-8")
        logger.info(f"Project logic patched with {len(new_rules)} new rules.")

    def get_lessons_context(self) -> str:
        """Retrieves existing lessons for prompt injection."""
        if not self.lessons_file.exists():
            return ""
        try:
            rules = json.loads(self.lessons_file.read_text(encoding="utf-8"))
            if not rules:
                return ""
            return "## Project-Specific Lessons (Historical)\n" + "\n".join(
                f"- {r}" for r in rules
            )
        except (json.JSONDecodeError, OSError):
            return ""


def _extract_json_array(response: str) -> List[str]:
    """Extract a JSON array from an LLM response (shared helper)."""
    return extract_json_array(response)
