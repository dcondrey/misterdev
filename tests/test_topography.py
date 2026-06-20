import tempfile
from pathlib import Path

from my_project_orchestrator.core.topography import (
    SymbolNode, SymbolGraph, TopographyEngine, _get_ts_parsers,
)


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
        node = SymbolNode("validate", "src/lib.py", "function", 1, 10, "def validate(): pass")
        engine.graph.symbols["src/lib.py:validate"] = node
        ctx = engine.get_context_for_task("check validation", ["src/lib.py"])
        assert "validate" in ctx
        assert "Topological Context" in ctx


def test_topography_max_symbols_cap():
    with tempfile.TemporaryDirectory() as td:
        engine = TopographyEngine(Path(td), None)
        engine._initialized = True
        for i in range(40):
            node = SymbolNode(f"fn_{i}", "big.py", "function", i, i+1, f"def fn_{i}(): pass")
            engine.graph.symbols[f"big.py:fn_{i}"] = node
        ctx = engine.get_context_for_task("query", ["big.py"], max_symbols=5)
        assert "omitted" in ctx
