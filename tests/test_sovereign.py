import tempfile
from pathlib import Path

from my_project_orchestrator.core.sovereign import (
    EphemeralCodeManager,
    RealTimeAligner,
    StrategyOptimizer,
    _extract_code_block,
)


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
