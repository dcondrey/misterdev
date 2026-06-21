import tempfile
from pathlib import Path

from my_project_orchestrator.core.topography import (
    SymbolNode,
    SymbolGraph,
    TopographyEngine,
)


def test_get_context_for_task_uses_ranker_over_cap():
    with tempfile.TemporaryDirectory() as td:
        engine = TopographyEngine(Path(td), llm_client=None)
        engine._initialized = True  # skip filesystem build; inject symbols
        for i in range(5):
            engine.graph.symbols[f"f.py:fn{i}"] = SymbolNode(
                f"fn{i}", "f.py", "function", i, i, f"def fn{i}(): pass"
            )

        class StubRanker:
            def __init__(self):
                self.seen = None

            def top_k(self, query, candidates, k):
                self.seen = (query, set(candidates), k)
                return ["f.py:fn3", "f.py:fn1"]

        ranker = StubRanker()
        out = engine.get_context_for_task(
            "do fn3 things", ["f.py"], max_symbols=2, ranker=ranker
        )
        # All 5 candidates offered to the ranker; only its picks rendered.
        assert ranker.seen[2] == 2
        assert len(ranker.seen[1]) == 5
        assert "fn3" in out and "fn1" in out
        assert "fn0" not in out


def test_get_context_for_task_no_ranker_keeps_arbitrary_slice():
    with tempfile.TemporaryDirectory() as td:
        engine = TopographyEngine(Path(td), llm_client=None)
        engine._initialized = True
        for i in range(5):
            engine.graph.symbols[f"f.py:fn{i}"] = SymbolNode(
                f"fn{i}", "f.py", "function", i, i, f"def fn{i}(): pass"
            )
        out = engine.get_context_for_task("q", ["f.py"], max_symbols=2)
        # Without a ranker, behavior is unchanged (a slice; omission note shown).
        assert "more symbols omitted" in out


def test_symbol_node_init():
    node = SymbolNode("foo", "src/main.py", "function", 10, 20, "def foo(): pass")
    assert node.name == "foo"
    assert node.file_path == "src/main.py"
    assert node.kind == "function"
    assert node.start_line == 10
    assert node.end_line == 20
    assert node.outgoing_calls == set()
    assert node.incoming_calls == set()


def test_symbol_node_repr():
    node = SymbolNode("bar", "lib.py", "class", 1, 50, "class bar: pass")
    assert "class" in repr(node)
    assert "bar" in repr(node)


def test_symbol_graph_empty_project():
    with tempfile.TemporaryDirectory() as td:
        graph = SymbolGraph(Path(td))
        graph.build()
        assert len(graph.symbols) == 0


def test_symbol_graph_skips_pycache():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cache_dir = td / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "cached.py").write_text("def cached(): pass\n")
        graph = SymbolGraph(td)
        graph.build()
        assert len(graph.symbols) == 0


def test_topography_engine_lazy_init():
    with tempfile.TemporaryDirectory() as td:
        engine = TopographyEngine(Path(td), None)
        assert not engine._initialized
        engine.initialize()
        assert engine._initialized


def test_topography_engine_no_reinit():
    with tempfile.TemporaryDirectory() as td:
        engine = TopographyEngine(Path(td), None)
        engine.initialize()
        engine.graph.symbols["fake"] = SymbolNode("f", "f.py", "function", 1, 1, "")
        engine.initialize()
        assert "fake" in engine.graph.symbols


def test_topography_engine_force_reinit():
    with tempfile.TemporaryDirectory() as td:
        engine = TopographyEngine(Path(td), None)
        engine.initialize()
        engine.graph.symbols["fake"] = SymbolNode("f", "f.py", "function", 1, 1, "")
        engine.initialize(force=True)
        assert "fake" not in engine.graph.symbols


def test_topography_get_context_empty():
    with tempfile.TemporaryDirectory() as td:
        engine = TopographyEngine(Path(td), None)
        ctx = engine.get_context_for_task("some query", ["nonexistent.py"])
        assert ctx == ""


def test_topography_get_context_with_symbols():
    with tempfile.TemporaryDirectory() as td:
        engine = TopographyEngine(Path(td), None)
        engine._initialized = True
        node = SymbolNode(
            "validate", "src/lib.py", "function", 1, 10, "def validate(): pass"
        )
        engine.graph.symbols["src/lib.py:validate"] = node
        ctx = engine.get_context_for_task("check validation", ["src/lib.py"])
        assert "validate" in ctx
        assert "Topological Context" in ctx


def test_topography_max_symbols_cap():
    with tempfile.TemporaryDirectory() as td:
        engine = TopographyEngine(Path(td), None)
        engine._initialized = True
        for i in range(40):
            node = SymbolNode(
                f"fn_{i}", "big.py", "function", i, i + 1, f"def fn_{i}(): pass"
            )
            engine.graph.symbols[f"big.py:fn_{i}"] = node
        ctx = engine.get_context_for_task("query", ["big.py"], max_symbols=5)
        assert "omitted" in ctx


def test_golden_excluded_from_symbol_graph():
    import tempfile
    from pathlib import Path
    from my_project_orchestrator.core.topography import SymbolGraph

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "tests" / "golden").mkdir(parents=True)
        (root / "src").mkdir()
        (root / "src" / "app.py").write_text(
            "def app_visible_symbol():\n    return 1\n", encoding="utf-8"
        )
        (root / "tests" / "golden" / "test_contract.py").write_text(
            "def golden_secret_symbol():\n    return 2\n", encoding="utf-8"
        )
        g = SymbolGraph(root, golden_paths=["tests/golden/"])
        g.build()
        indexed = " ".join(g.symbols.keys())
        # Golden symbols must never be indexed (regardless of parser availability).
        assert "golden_secret_symbol" not in indexed
