"""ToolCorpus + promote_from_corpus: the two-timescale loop closure.

Invented tools accumulate per-task outcomes for free; promotion admits only those
whose association with success holds on held-out tasks. Pure/offline.
"""

from misterdev.core.evolution.tool_corpus import (
    ToolCorpus,
    promote_from_corpus,
)
from misterdev.core.evolution.tool_library import ToolLibrary


def test_missing_file_degrades_to_empty(tmp_path):
    c = ToolCorpus(tmp_path / "tools.json")
    assert c.records() == []
    assert c.stats() == {"tools": 0, "observations": 0}


def test_record_accumulates_per_task_outcomes(tmp_path):
    c = ToolCorpus(tmp_path / "tools.json")
    src = "print('inverse')"
    c.record(src, "python", "task-a", True)
    c.record(src, "python", "task-b", False)
    recs = c.records()
    assert len(recs) == 1
    assert recs[0].outcomes == {"task-a": True, "task-b": False}
    assert c.stats() == {"tools": 1, "observations": 2}


def test_same_source_collapses_across_whitespace(tmp_path):
    c = ToolCorpus(tmp_path / "tools.json")
    c.record("print('x')  ", "python", "t1", True)
    c.record("print('x')", "python", "t2", True)  # trailing-space variant
    assert len(c.records()) == 1  # one tool, two observations


def test_rerun_of_same_task_updates_not_double_counts(tmp_path):
    c = ToolCorpus(tmp_path / "tools.json")
    c.record("print('x')", "python", "t1", False)
    c.record("print('x')", "python", "t1", True)  # same task, better outcome
    assert c.records()[0].outcomes == {"t1": True}


def test_persists_across_instances(tmp_path):
    p = tmp_path / "tools.json"
    ToolCorpus(p).record("print('x')", "python", "t1", True)
    assert len(ToolCorpus(p).records()) == 1


def test_promote_admits_a_generalizing_tool(tmp_path):
    # A tool present on many tasks that mostly resolve, above a low baseline,
    # holding on the held-out split -> promoted.
    c = ToolCorpus(tmp_path / "tools.json")
    for i in range(12):
        c.record("print('helpful')", "python", f"t{i}", resolved=(i % 6 != 0))  # ~83%
    lib = ToolLibrary(tmp_path / "lib.json")
    promoted = promote_from_corpus(c, lib, baseline_rate=0.4, min_observations=5)
    assert promoted  # generalizes above the 40% baseline on both pools
    assert [t.niche for t in lib.elites()] == ["python"]


def test_promote_rejects_below_minimum_observations(tmp_path):
    c = ToolCorpus(tmp_path / "tools.json")
    for i in range(3):
        c.record("print('rare')", "python", f"t{i}", True)
    lib = ToolLibrary(tmp_path / "lib.json")
    assert promote_from_corpus(c, lib, baseline_rate=0.4, min_observations=5) == []
    assert lib.elites() == []


def test_promote_rejects_a_tool_that_does_not_beat_baseline(tmp_path):
    # Tool resolves ~40% of its tasks; baseline is already 60% -> no gain -> rejected.
    c = ToolCorpus(tmp_path / "tools.json")
    for i in range(12):
        c.record("print('meh')", "python", f"t{i}", resolved=(i % 5 == 0))  # 20%
    lib = ToolLibrary(tmp_path / "lib.json")
    assert promote_from_corpus(c, lib, baseline_rate=0.6, min_observations=5) == []
    assert lib.elites() == []
