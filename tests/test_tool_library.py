"""ToolLibrary: the consolidation half of two-timescale evolution.

A self-authored tool is admitted only if it GENERALIZES (held-out gate) and beats
its capability niche's incumbent. Promoted tools seed future runs — the memory
live-SWE-agent lacks. All pure/offline, no benchmark, no code execution.
"""

from pathlib import Path

from misterdev.core.evolution.fitness import FitnessScore
from misterdev.core.evolution.tool_library import ToolCandidate, ToolLibrary


def _tool(tid, niche, resolved, total, cost=1.0, regressions=0):
    return ToolCandidate(
        id=tid,
        niche=niche,
        source=f"# tool {tid}\n",
        resolved=resolved,
        total=total,
        cost=cost,
        regressions=regressions,
        provenance="task-x",
    )


def _score(resolved, total, cost=1.0, regressions=0):
    return FitnessScore(resolved, total, cost, regressions)


def test_missing_file_degrades_to_empty(tmp_path):
    lib = ToolLibrary(tmp_path / "tools.json")
    assert lib.elites() == []
    assert lib.seed() == []
    assert lib.elite("anything") is None


def test_corrupt_file_degrades_to_empty(tmp_path):
    p = tmp_path / "tools.json"
    p.write_text("{ not json", encoding="utf-8")
    assert ToolLibrary(p).elites() == []


def test_generalizing_tool_is_admitted_and_seeds(tmp_path):
    lib = ToolLibrary(tmp_path / "tools.json")
    # Gains on DERIVE (6/10 -> 8/10) and does not drop HOLDOUT (5/10 held).
    decision = lib.consider(
        _tool("t1", "reproduce-pytest", 8, 10),
        derive=_score(8, 10),
        derive_base=_score(6, 10),
        holdout=_score(5, 10),
        holdout_base=_score(5, 10),
    )
    assert decision.promote
    assert [t.id for t in lib.elites()] == ["t1"]
    # A fresh run seeds from it: accumulated capability, not reinvented.
    assert [t.id for t in lib.seed()] == ["t1"]


def test_overfit_tool_is_rejected(tmp_path):
    lib = ToolLibrary(tmp_path / "tools.json")
    # Big DERIVE gain (6->9) but HOLDOUT drops (6->4): bought derive-specific score
    # at the cost of general capability — the exact failure the gate must catch.
    decision = lib.consider(
        _tool("t2", "reproduce-pytest", 9, 10),
        derive=_score(9, 10),
        derive_base=_score(6, 10),
        holdout=_score(4, 10),
        holdout_base=_score(6, 10),
    )
    assert not decision.promote
    assert "OVERFIT" in decision.reason
    # Rejected: never enters the library, so it can never seed a future run.
    assert lib.elites() == []
    assert lib.seed() == []


def test_regression_blocks_admission(tmp_path):
    lib = ToolLibrary(tmp_path / "tools.json")
    decision = lib.consider(
        _tool("t3", "edit-tool", 9, 10, regressions=1),
        derive=_score(9, 10, regressions=1),
        derive_base=_score(6, 10),
        holdout=_score(7, 10),
        holdout_base=_score(6, 10),
    )
    assert not decision.promote
    assert lib.elites() == []


def test_better_tool_replaces_niche_incumbent(tmp_path):
    lib = ToolLibrary(tmp_path / "tools.json")
    lib.consider(
        _tool("t1", "reproduce-pytest", 7, 10),
        derive=_score(7, 10),
        derive_base=_score(6, 10),
        holdout=_score(5, 10),
        holdout_base=_score(5, 10),
    )
    # A stronger generalizing tool for the SAME niche takes the slot.
    lib.consider(
        _tool("t2", "reproduce-pytest", 9, 10),
        derive=_score(9, 10),
        derive_base=_score(6, 10),
        holdout=_score(6, 10),
        holdout_base=_score(5, 10),
    )
    elites = lib.elites()
    assert [t.id for t in elites] == ["t2"]


def test_generalizing_but_not_beating_incumbent_is_not_admitted(tmp_path):
    lib = ToolLibrary(tmp_path / "tools.json")
    lib.consider(
        _tool("strong", "edit-tool", 9, 10),
        derive=_score(9, 10),
        derive_base=_score(6, 10),
        holdout=_score(6, 10),
        holdout_base=_score(5, 10),
    )
    # A weaker (but still generalizing) tool for the same niche must not displace
    # the stronger incumbent, even though it passes the held-out gate on its own.
    decision = lib.consider(
        _tool("weak", "edit-tool", 7, 10),
        derive=_score(7, 10),
        derive_base=_score(6, 10),
        holdout=_score(6, 10),
        holdout_base=_score(5, 10),
    )
    assert not decision.promote
    assert "does not beat" in decision.reason
    assert [t.id for t in lib.elites()] == ["strong"]


def test_distinct_niches_coexist_and_seed_ranked(tmp_path):
    lib = ToolLibrary(tmp_path / "tools.json")
    lib.consider(
        _tool("weakish", "niche-a", 7, 10),
        derive=_score(7, 10),
        derive_base=_score(6, 10),
        holdout=_score(5, 10),
        holdout_base=_score(5, 10),
    )
    lib.consider(
        _tool("strongish", "niche-b", 9, 10),
        derive=_score(9, 10),
        derive_base=_score(6, 10),
        holdout=_score(5, 10),
        holdout_base=_score(5, 10),
    )
    # Both niches kept (MAP-Elites), and seed() ranks the globally stronger first.
    assert {t.niche for t in lib.elites()} == {"niche-a", "niche-b"}
    assert [t.id for t in lib.seed()] == ["strongish", "weakish"]
    assert [t.id for t in lib.seed(limit=1)] == ["strongish"]


def test_persists_across_instances(tmp_path):
    p = tmp_path / "tools.json"
    ToolLibrary(p).consider(
        _tool("t1", "reproduce-pytest", 8, 10),
        derive=_score(8, 10),
        derive_base=_score(6, 10),
        holdout=_score(5, 10),
        holdout_base=_score(5, 10),
    )
    # A new library over the same path sees the admitted tool.
    assert [t.id for t in ToolLibrary(p).elites()] == ["t1"]
