"""T4.2 — decomposed tasks are checked against a size/verifiability invariant.

`enforce_task_invariants` flags a task that touches too many files (size) or carries
a blank/placeholder acceptance criterion (verifiability), recording the reason on
processor_data so downstream can split or reject it. A well-formed task is untouched.
"""

from misterdev.core.models import Task
from misterdev.core.planning.decomposer import enforce_task_invariants


def _task(tid, files=None, acceptance="pytest tests/test_x.py passes"):
    return Task(
        id=tid,
        description="do a thing",
        project_ref=".",
        files_to_modify=list(files or ["a.py"]),
        acceptance_criteria=acceptance,
    )


def _violations(task):
    return (task.processor_data or {}).get("invariant_violations") or []


def test_wellformed_task_has_no_violations():
    t = _task("T1")
    enforce_task_invariants([t])
    assert _violations(t) == []


def test_oversized_task_flagged():
    t = _task("T2", files=[f"f{i}.py" for i in range(25)])
    enforce_task_invariants([t], max_files=20)
    assert any("file" in v.lower() for v in _violations(t))


def test_blank_acceptance_flagged():
    t = _task("T3", acceptance="")
    enforce_task_invariants([t])
    assert any("accept" in v.lower() or "verif" in v.lower() for v in _violations(t))


def test_placeholder_acceptance_flagged():
    t = _task("T4", acceptance="works")
    enforce_task_invariants([t])
    assert _violations(t)


def test_returns_the_task_list():
    tasks = [_task("T5")]
    assert enforce_task_invariants(tasks) is tasks
