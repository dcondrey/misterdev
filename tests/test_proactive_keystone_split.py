"""T4.3 — a keystone (high-fan-in) task is split into chained sub-units.

Splitting a task many others depend on reduces per-attempt blast radius. The split
partitions its files across chained sub-units and rewires every dependent to the FINAL
sub-unit, so the dependency ordering is preserved. Non-keystones and keystones with too
few files to partition are unchanged.
"""

from misterdev.core.models import Task
from misterdev.core.planning.decomposer import split_keystone_tasks


def _task(tid, deps=None, create=None, modify=None):
    return Task(
        id=tid,
        description="x",
        project_ref=".",
        dependencies=list(deps or []),
        files_to_create=list(create or []),
        files_to_modify=list(modify or []),
        acceptance_criteria="pytest passes",
    )


def _keystone_plan(n_deps=3, files=("a.py", "b.py")):
    key = _task("K", modify=list(files))
    deps = [_task(f"D{i}", deps=["K"]) for i in range(n_deps)]
    return [key] + deps


def test_keystone_is_split_and_dependents_rewired():
    tasks = _keystone_plan(n_deps=3, files=("a.py", "b.py"))
    out = split_keystone_tasks(tasks, fanin_threshold=3, min_units=2)
    ids = {t.id for t in out}
    assert "K" not in ids  # replaced by sub-units
    subs = sorted(t.id for t in out if t.id.startswith("K-part"))
    assert len(subs) == 2
    # Later sub-unit chains on the earlier one.
    part2 = next(t for t in out if t.id == "K-part2")
    assert "K-part1" in part2.dependencies
    # Every original dependent now waits on the FINAL sub-unit, not "K".
    final = subs[-1]
    for d in [t for t in out if t.id.startswith("D")]:
        assert final in d.dependencies and "K" not in d.dependencies
    # Files were partitioned across the units (no loss, no duplication).
    part1 = next(t for t in out if t.id == "K-part1")
    assert set(part1.files_to_modify) | set(part2.files_to_modify) == {"a.py", "b.py"}
    assert not (set(part1.files_to_modify) & set(part2.files_to_modify))


def test_low_fanin_task_unchanged():
    tasks = _keystone_plan(n_deps=1, files=("a.py", "b.py"))
    out = split_keystone_tasks(tasks, fanin_threshold=3)
    assert {t.id for t in out} == {"K", "D0"}


def test_keystone_with_one_file_not_split():
    tasks = _keystone_plan(n_deps=5, files=("only.py",))
    out = split_keystone_tasks(tasks, fanin_threshold=3, min_units=2)
    assert "K" in {t.id for t in out}


def test_completed_keystone_not_re_split():
    tasks = _keystone_plan(n_deps=3, files=("a.py", "b.py"))
    out = split_keystone_tasks(
        tasks, fanin_threshold=3, min_units=2, completed_ids=frozenset({"K"})
    )
    # "K" is already done — must not be replaced by sub-units.
    assert "K" in {t.id for t in out}
    assert not any(t.id.startswith("K-part") for t in out)
