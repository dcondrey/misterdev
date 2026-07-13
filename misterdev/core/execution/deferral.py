"""The parked-task question book: the walk-away interface.

When a run parks tasks that need human input (a missing credential, a judgment
call), it writes them here as ``.orchestrator/QUESTIONS.md`` — a plain checklist
the user fills in on their own time — plus an append-only ``deferrals.jsonl``
record. A follow-up run reads the answers back and resumes only the answered
tasks, so the loop is: run → walk away → answer the questions → re-run → done.

Answers PERSIST across runs: rewriting the book preserves any answer the user has
already typed for a task that is still parked, so re-running never wipes their
input. Everything is best-effort file I/O — a broken book never fails a build.
"""

import json
from pathlib import Path
from typing import Dict, List

from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)

_PLACEHOLDER = "_(write your answer here)_"
_HEADER = (
    "# Questions from misterdev — answer inline, then re-run\n\n"
    "These tasks are PARKED waiting on you. Replace each **Answer:** placeholder "
    "with your answer, then re-run the same `misterdev run --tasks ...` command — "
    "answered tasks resume, unanswered ones stay parked. For a missing "
    "credential, just provide it in your environment and re-run (no text needed).\n"
)


class DeferralBook:
    """Reads/writes the parked-task questions for a project."""

    def __init__(self, orchestrator_dir: Path):
        self.dir = Path(orchestrator_dir)
        self.md_path = self.dir / "QUESTIONS.md"
        self.log_path = self.dir / "deferrals.jsonl"

    def load_answers(self) -> Dict[str, str]:
        """``{task_id: answer}`` for every task whose Answer line the user filled in.

        Parses the ``## <task_id> — ...`` sections and their ``- Answer:`` line.
        A line still holding the placeholder (or empty) is treated as unanswered.
        """
        answers: Dict[str, str] = {}
        if not self.md_path.exists():
            return answers
        try:
            lines = self.md_path.read_text(encoding="utf-8").splitlines()
        except OSError as e:
            logger.warning(f"QUESTIONS.md unreadable ({e}); no answers loaded.")
            return answers
        current: str = ""
        for line in lines:
            if line.startswith("## "):
                current = line[3:].split("—", 1)[0].split(" - ", 1)[0].strip()
                continue
            low = line.strip()
            marker = "- answer:"
            if current and low.lower().startswith(marker):
                ans = line.strip()[len(marker) :].strip().strip("*_ ")
                if ans and ans != _PLACEHOLDER.strip("*_ "):
                    answers[current] = ans
        return answers

    def write(self, deferrals: List[dict]) -> None:
        """Write QUESTIONS.md for the currently-parked tasks, preserving any answers
        already typed for tasks that are still parked. Appends machine records to
        deferrals.jsonl. ``deferrals`` items: {id, title, reason, questions:[...]}.
        """
        if not deferrals:
            return
        existing = self.load_answers()
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            blocks = [_HEADER]
            for d in deferrals:
                tid = str(d.get("id", "")).strip()
                title = str(d.get("title", "") or "").strip()
                reason = str(d.get("reason", "") or "").strip()
                questions = d.get("questions") or []
                answer = existing.get(tid, _PLACEHOLDER)
                head = f"## {tid}" + (f" — {title}" if title else "")
                lines = [head]
                if reason:
                    lines.append(f"- Reason: {reason}")
                for q in questions:
                    lines.append(f"- Question: {q}")
                lines.append(f"- Answer: {answer}")
                blocks.append("\n".join(lines))
            self.md_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
            with self.log_path.open("a", encoding="utf-8") as fh:
                for d in deferrals:
                    fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        except OSError as e:  # the question book must never sink a run
            logger.warning(f"Could not write the deferral question book ({e}).")
