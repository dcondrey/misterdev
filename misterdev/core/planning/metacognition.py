"""Metacognitive Trace Auditing and Project-Specific Learning.

Reviews build session history to generate project-specific rules, stored in a
scored, self-reinforcing memory (see :mod:`.lesson_store`) so misterdev gets
smarter each run without forgetting the lessons that keep proving useful.
"""

from pathlib import Path
from typing import List

from misterdev.core.planning.lesson_store import _MAX_LESSONS, LessonStore
from misterdev.llm.client import BaseLLMClient
from misterdev.llm.responses import extract_json_array
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)

__all__ = ["SessionAuditor", "_extract_json_array", "_MAX_LESSONS"]


class SessionAuditor:
    """Audits task execution traces to extract and reinforce 'Lessons Learned'."""

    def __init__(self, project_path: Path, llm_client: BaseLLMClient, embedder=None):
        self.project_path = Path(project_path)
        self.llm = llm_client
        self.store = LessonStore(
            self.project_path / ".orchestrator" / "lessons.json", embedder=embedder
        )
        # Ids of the lessons injected into THIS build, captured at retrieval so the
        # end-of-run audit can credit exactly those with the run's outcome.
        self._injected_ids: List[int] = []

    @property
    def lessons_file(self) -> Path:
        """Path to the backing store (kept for callers that reference it)."""
        return self.store.path

    def audit_session(
        self,
        completed_tasks: List,
        failed_tasks: List,
        scratchpad_content: str,
    ) -> str:
        """Run the audit and fold any discovered rules into the scored store."""
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
        result = "No new lessons learned."
        try:
            response = self.llm.generate_code(
                prompt, "You are a senior project auditor."
            )
            new_rules = _extract_json_array(response)
            if new_rules:
                self.store.record(new_rules)
                result = "\n".join(f"- {r}" for r in new_rules)
        except Exception as e:
            logger.error(f"Metacognitive audit failed: {e}")

        # Credit the lessons injected into this build with its measured outcome, so
        # future retrieval reinforces what actually helped rather than what merely
        # recurred. Runs with no attempted tasks carry no signal and are skipped.
        try:
            attempted = len(completed_tasks) + len(failed_tasks)
            if attempted and self._injected_ids:
                outcome = len(completed_tasks) / attempted
                self.store.credit(self._injected_ids, outcome)
        except Exception as e:
            logger.warning(f"Lesson efficacy crediting failed (non-fatal): {e}")

        return result

    def _save_lessons(self, new_rules: List) -> int:
        """Fold rules into the scored store (kept as a thin, direct entry point)."""
        return self.store.record(new_rules)

    def get_lessons_context(self, query: str = "") -> str:
        """Retrieve the most relevant, highest-value lessons for prompt injection.

        ``query`` (typically the build goal) biases retrieval toward lessons that
        pertain to the work at hand; empty ranks by proven value alone.
        """
        injected = self.store.retrieve_lessons(query)
        self._injected_ids = [le.id for le in injected]
        if not injected:
            return ""
        return "## Project-Specific Lessons (Historical)\n" + "\n".join(
            f"- {le.text}" for le in injected
        )


def _extract_json_array(response: str) -> List[str]:
    """Extract a JSON array from an LLM response (shared helper)."""
    return extract_json_array(response)
