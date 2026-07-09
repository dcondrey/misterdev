"""A no-usable-edit response is a formatting failure, not a solve attempt, so it
must not consume the solve budget.

Regression for the observed polyglot failures where attempt 1 was burned by a
free model that 429'd / returned no edit, leaving too few real attempts. With a
single configured attempt, a no-edit response followed by a valid fix must still
succeed — the no-output response buys a bounded extra iteration.
"""

import json

from misterdev.config import DEFAULT_CONFIG
from misterdev.core.execution.project import Project
from misterdev.core.models import Task
from misterdev.llm.client import LLMResponse, LLMUsage
from misterdev.task_executors.markdown_plan_executor import MarkdownPlanExecutor
from tests.test_llm_client import FakeLLMClient


def _run(tmp_path, monkeypatch, first_response: str):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    (tmp_path / "mod.py").write_text("def answer():\n    return 0\n")
    (tmp_path / "mod_test.py").write_text(
        "from mod import answer\n\n\ndef test_answer():\n    assert answer() == 42\n"
    )
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg.update(
        {
            "name": "f",
            "build_command": "true",
            "language": "python",
            "test_command": "python -m pytest -q",
        }
    )
    # One configured attempt, and start on "surgical" so the post-loop strategy
    # escalation never fires — this isolates the in-loop attempt budget.
    cfg["orchestrator"]["max_task_attempts"] = 1
    # Static routing (no ledger selection) so the test is deterministic.
    cfg["llm"]["dynamic_selection"] = False
    cfg["llm"]["escalation"] = []
    project = Project(tmp_path, cfg)

    fix = "```python:mod.py\ndef answer():\n    return 42\n```\n"

    class _Seq(FakeLLMClient):
        def __init__(self):
            super().__init__(responses=[])
            self.calls = 0

        def _call(self, prompt, system_prompt):
            self.calls += 1
            # First response yields NO applicable edit (plain prose, no fenced
            # file path); the second is the real fix.
            content = first_response if self.calls == 1 else fix
            return LLMResponse(content=content, usage=LLMUsage())

    client = _Seq()
    project.llm_client = client
    task = Task(
        id="T-1", description="make answer() return 42", project_ref=str(tmp_path)
    )
    task.acceptance_criteria = "answer() returns 42"
    task.files_to_modify = ["mod.py"]
    task.processor_data["strategy"] = "surgical"
    result = MarkdownPlanExecutor().execute(task, project, use_git_branch=False)
    return client, result, (tmp_path / "mod.py").read_text()


def test_no_edit_response_does_not_burn_the_only_attempt(tmp_path, monkeypatch):
    # Attempt 1: no applicable edit (prose). Attempt 2 (granted, not charged):
    # the real fix. With a single configured attempt this only passes because the
    # no-output response did not consume the solve budget.
    client, _result, final = _run(tmp_path, monkeypatch, "I will now fix the function.")
    assert client.calls >= 2  # the fix response was actually reached
    assert "return 42" in final  # the fix landed


def test_unapplied_search_replace_does_not_burn_the_only_attempt(tmp_path, monkeypatch):
    # An anchored SEARCH/REPLACE that matches nothing applies no change — also a
    # no-output response that must not cost the attempt.
    bad_hunk = (
        "```python:mod.py\n<<<<<<< SEARCH\ndef nonexistent_anchor():\n"
        "=======\ndef answer():\n    return 42\n>>>>>>> REPLACE\n```\n"
    )
    client, _result, final = _run(tmp_path, monkeypatch, bad_hunk)
    assert client.calls >= 2
    assert "return 42" in final
