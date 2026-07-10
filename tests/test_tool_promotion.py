"""tool_promotion: the deliberate pass that closes the two-timescale loop.

Derives the without-tool baseline PER NICHE from the reproduction corpus and
admits generalizing tools into the library. Pure/offline.
"""

from types import SimpleNamespace

from misterdev.core.evolution.tool_corpus import ToolCorpus
from misterdev.core.evolution.tool_library import ToolLibrary
from misterdev.core.evolution.tool_promotion import run_tool_promotion
from misterdev.core.learning.reproduction import ReproductionCorpus


def _r(name, language, resolved):
    return SimpleNamespace(
        name=name, language=language, resolved=resolved, error="", output=""
    )


def test_reproduction_resolve_rate_baseline(tmp_path):
    repro = ReproductionCorpus(tmp_path / "r.json")
    repro.update(
        [_r("a", "python", True), _r("b", "python", False), _r("c", "rust", True)]
    )
    assert repro.resolve_rate() == 2 / 3  # global
    assert repro.resolve_rate("python") == 0.5
    assert repro.resolve_rate("rust") == 1.0
    assert repro.resolve_rate("go") is None  # no data -> caller uses a prior


def _seed(tmp_path, repro_results, tool_outcomes):
    ev = tmp_path / ".orchestrator" / "evolution"
    ev.mkdir(parents=True, exist_ok=True)
    ReproductionCorpus(ev / "reproduction.json").update(repro_results)
    corpus = ToolCorpus(ev / "tool_corpus.json")
    for tid, resolved in tool_outcomes:
        corpus.record("print('helpful')", "python", tid, resolved)
    return ev


def test_promotes_a_tool_that_beats_its_niche_baseline(tmp_path):
    # python baseline ~30% (from reproduction corpus); tool resolves ~75% -> promote.
    repro = [_r(f"c{i}", "python", i < 3) for i in range(10)]  # 30%
    tools = [(f"t{i}", i % 4 != 0) for i in range(12)]  # 75%
    ev = _seed(tmp_path, repro, tools)
    res = run_tool_promotion(tmp_path, min_observations=5)
    assert res["promoted"]
    assert len(ToolLibrary(ev / "tool_library.json").elites()) == 1


def test_rejects_a_tool_when_niche_baseline_already_high(tmp_path):
    # python baseline 100%; tool only 50% -> no gain over baseline -> rejected.
    repro = [_r(f"c{i}", "python", True) for i in range(10)]  # 100%
    tools = [(f"t{i}", i % 2 == 0) for i in range(12)]  # 50%
    _seed(tmp_path, repro, tools)
    res = run_tool_promotion(tmp_path, min_observations=5)
    assert res["promoted"] == []


def test_uses_default_baseline_for_an_unknown_niche(tmp_path):
    # No reproduction data for the niche -> falls back to default_baseline, which
    # a strong tool still beats.
    ev = tmp_path / ".orchestrator" / "evolution"
    ev.mkdir(parents=True)
    corpus = ToolCorpus(ev / "tool_corpus.json")
    for i in range(12):
        corpus.record("print('x')", "python", f"t{i}", i % 6 != 0)  # ~83%
    res = run_tool_promotion(tmp_path, default_baseline=0.3, min_observations=5)
    assert res["promoted"]


def test_missing_corpora_degrades_cleanly(tmp_path):
    res = run_tool_promotion(tmp_path)
    assert res["promoted"] == []
    assert res["library_size"] == 0
    assert res["corpus"] == {"tools": 0, "observations": 0}
