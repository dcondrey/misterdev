"""T4.1 — decompose-rung sub-steps are EXECUTED as real child tasks.

`_escalate_decompose` used to only record sub-steps and park the task. It now runs
each sub-step through the real per-task gate via `self.execute`, one recursion level
deep (a sub-step can never itself re-decompose). If all sub-steps complete, the parent
is satisfied; at depth >= 1 it falls back to deferring.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from misterdev.core.models import ExecutionResult, Task
from misterdev.task_executors.markdown_plan_executor import MarkdownPlanExecutor


class _Spy(MarkdownPlanExecutor):
    def __init__(self, child_status="completed"):
        self.execute_calls = []
        self._child_status = child_status

    def execute(self, task, project, use_git_branch=True, _depth=0):
        self.execute_calls.append((task.id, _depth))
        return ExecutionResult(status=self._child_status, message="")

    def _record_invented_tools(self, project, task, resolved):
        pass


def _parent():
    return Task(
        id="BIG",
        description="big task",
        project_ref=".",
        acceptance_criteria="Do part one thoroughly\nDo part two thoroughly",
        files_to_modify=["a.py"],
        processor_data={},
    )


def _project():
    return SimpleNamespace(task_manager=MagicMock(), topography=None)


def test_substeps_execute_as_real_children():
    ex = _Spy()
    result = ex._escalate_decompose(_project(), _parent(), "")
    # Two acceptance lines -> two sub-steps, each executed as a child at depth+1.
    assert len(ex.execute_calls) == 2
    assert all(d == 1 for _, d in ex.execute_calls)
    assert result.status == "completed"


def test_failed_substep_defers_parent():
    ex = _Spy(child_status="failed")
    result = ex._escalate_decompose(_project(), _parent(), "")
    assert len(ex.execute_calls) == 2
    assert result.status == "deferred"


def test_no_recursion_at_depth_one():
    ex = _Spy()
    result = ex._escalate_decompose(_project(), _parent(), "", _depth=1)
    assert ex.execute_calls == []  # deferred, not executed
    assert result.status == "deferred"
