import tempfile
from pathlib import Path

from my_project_orchestrator.core.planning.sovereign import (
    ABMCTSPlanner,
    EphemeralCodeManager,
    ProbeGenerator,
    RealTimeAligner,
    StrategyOptimizer,
    ToolSynthesizer,
    _extract_code_block,
)


class _FakeLLM:
    def __init__(self, responses=None, raises=False):
        self.responses = list(responses or [])
        self.raises = raises
        self.calls = 0

    def generate_code(self, prompt, system_prompt=""):
        self.calls += 1
        if self.raises:
            raise RuntimeError("llm down")
        return self.responses.pop(0) if self.responses else "x"


def test_abmcts_skips_branching_for_short_spec():
    llm = _FakeLLM()
    planner = ABMCTSPlanner(llm)
    out = planner.branch_and_evaluate("short task", "ctx")
    assert out == "short task" and llm.calls == 0  # no LLM calls


def test_abmcts_branches_and_selects_for_long_spec():
    long_task = "word " * 60
    llm = _FakeLLM(responses=["path A", "path B", "path C", "WINNER"])
    out = ABMCTSPlanner(llm).branch_and_evaluate(long_task, "ctx", branches=3)
    assert out == "WINNER"
    assert llm.calls == 4  # 3 branches + 1 evaluation


def test_probe_generator_parses_and_handles_failure():
    llm = _FakeLLM(responses=['[{"name": "p", "purpose": "x", "script": "print(1)"}]'])
    probes = ProbeGenerator(llm).generate_probes("spec", "summary")
    assert len(probes) == 1 and probes[0]["name"] == "p"
    # LLM failure -> empty list, never raises
    assert ProbeGenerator(_FakeLLM(raises=True)).generate_probes("s", "a") == []


def test_tool_synthesizer_writes_tool_file():
    with tempfile.TemporaryDirectory() as td:
        llm = _FakeLLM(responses=["```python\nprint('tool')\n```"])
        path = ToolSynthesizer(Path(td)).synthesize_tool("My Tool", "do stuff", llm)
        assert Path(path).exists()
        assert "print('tool')" in Path(path).read_text()


def test_strategy_optimizer_selects_then_caches():
    llm = _FakeLLM(responses=["surgical"])
    opt = StrategyOptimizer()
    first = opt.select_best_strategy("desc", "feature", "proj", llm)
    assert first == "surgical" and llm.calls == 1
    # second call for same category is served from cache (no new LLM call)
    second = opt.select_best_strategy("desc2", "feature", "proj", llm)
    assert second == "surgical" and llm.calls == 1


def test_strategy_optimizer_invalid_falls_back_to_iterative():
    llm = _FakeLLM(responses=["nonsense-strategy"])
    out = StrategyOptimizer().select_best_strategy("d", "bugfix", "p", llm)
    assert out == "iterative"


def test_ephemeral_runs_script():
    with tempfile.TemporaryDirectory() as td:
        ecm = EphemeralCodeManager(Path(td))
        success, output = ecm.run_ephemeral_script("print('hello')")
        assert success
        assert "hello" in output
        ecm.cleanup()


def test_ephemeral_captures_failure():
    with tempfile.TemporaryDirectory() as td:
        ecm = EphemeralCodeManager(Path(td))
        success, output = ecm.run_ephemeral_script("raise ValueError('boom')")
        assert not success
        assert "boom" in output
        ecm.cleanup()


def test_ephemeral_cleanup():
    with tempfile.TemporaryDirectory() as td:
        ecm = EphemeralCodeManager(Path(td))
        ecm.run_ephemeral_script("print(1)")
        assert ecm.ephemeral_dir.exists()
        ecm.cleanup()
        assert not ecm.ephemeral_dir.exists()


def test_realtime_aligner_certify():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        aligner = RealTimeAligner(td)
        aligner.certify_decision("Use parking_lot", "Better performance")
        assert len(aligner.data["decisions"]) == 1
        assert aligner.data["decisions"][0]["decision"] == "Use parking_lot"


def test_realtime_aligner_consensus_context():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        aligner = RealTimeAligner(td)
        assert "No prior decisions" in aligner.get_consensus_context()
        aligner.certify_decision("Use FxHashMap", "Faster hashing")
        ctx = aligner.get_consensus_context()
        assert "FxHashMap" in ctx
        assert "Certified" in ctx


def test_realtime_aligner_persistence():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        a1 = RealTimeAligner(td)
        a1.certify_decision("decision1", "reason1")
        a2 = RealTimeAligner(td)
        assert len(a2.data["decisions"]) == 1
        assert a2.data["decisions"][0]["decision"] == "decision1"


def test_realtime_aligner_corrupt_file():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cert_file = td / ".orchestrator" / "consensus.json"
        cert_file.parent.mkdir(parents=True, exist_ok=True)
        cert_file.write_text("not json")
        aligner = RealTimeAligner(td)
        assert aligner.data == {"invariants": [], "decisions": []}


def test_strategy_optimizer_cache():
    optimizer = StrategyOptimizer()
    optimizer._cache["feature"] = "iterative"
    assert optimizer._cache["feature"] == "iterative"


def test_strategy_optimizer_strategies():
    assert "surgical" in StrategyOptimizer.STRATEGIES
    assert "iterative" in StrategyOptimizer.STRATEGIES
    assert "architectural" in StrategyOptimizer.STRATEGIES
    assert "agentic" in StrategyOptimizer.STRATEGIES


def test_extract_code_block_python():
    response = "Here is the code:\n```python\nx = 1\ny = 2\n```\nDone."
    assert _extract_code_block(response) == "x = 1\ny = 2"


def test_extract_code_block_no_lang():
    response = "Code:\n```\nfn main() {}\n```"
    assert _extract_code_block(response) == "fn main() {}"


def test_extract_code_block_no_fence():
    response = "x = 1\ny = 2"
    assert _extract_code_block(response) == "x = 1\ny = 2"


def test_extract_code_block_multiple():
    response = "First:\n```\nblock1\n```\nSecond:\n```\nblock2\n```"
    assert _extract_code_block(response) == "block1"
