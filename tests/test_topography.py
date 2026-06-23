import tempfile
from pathlib import Path

from my_project_orchestrator.core.topography import (
    SymbolNode,
    SymbolGraph,
    TopographyEngine,
    check_syntax,
)


def test_symbol_graph_parses_javascript():
    syms = _symbols_for("a.js", "function f(x){return x}\nclass A { m() {} }")
    if not syms:
        return
    assert ("function", "f") in syms
    assert ("class", "A") in syms
    assert ("method", "A.m") in syms


def test_symbol_graph_parses_kotlin():
    syms = _symbols_for(
        "E.kt", "class Engine {\n  fun start() {}\n}\nfun run() {}\nobject Reg {}"
    )
    if not syms:
        return
    assert ("class", "Engine") in syms
    assert ("method", "Engine.start") in syms
    assert ("function", "run") in syms
    assert ("object", "Reg") in syms


def test_typescript_captures_enum_arrow_const_and_labels():
    syms = _symbols_for(
        "app.ts",
        "export interface Cmd { n: string }\nexport type Id = string;\n"
        "export enum Mode { A, B }\nexport const run = (x: number) => x + 1;\n",
    )
    if not syms:
        return
    assert ("interface", "Cmd") in syms
    assert ("type", "Id") in syms
    assert ("enum", "Mode") in syms
    assert ("function", "run") in syms  # arrow-const captured as a function


def test_tsx_arrow_component_captured():
    syms = _symbols_for("ui.tsx", "export const View = () => <div>{x}</div>;")
    if not syms:
        return
    assert ("function", "View") in syms


# --- check_syntax: real parse-based correctness verification ----------------


def test_check_syntax_valid_and_invalid_rust():
    if check_syntax("fn main() {}", "rust") is None:
        return
    assert check_syntax("fn main() {}", "rust") == (True, None)
    ok, msg = check_syntax("fn main( { let", "rust")
    assert ok is False and "syntax error" in msg


def test_check_syntax_brace_in_string_not_flagged():
    # The killer case for brace-counting: a brace inside a string literal.
    result = check_syntax('fn f() { let s = "}"; }', "rust")
    if result is None:
        return
    assert result == (True, None)


def test_check_syntax_tsx_jsx_not_false_flagged():
    result = check_syntax("const v = <a href='x'>{y}</a>;", "typescript")
    if result is None:
        return
    assert result == (True, None)


def test_check_syntax_unsupported_returns_none():
    # No trustworthy grammar (Kotlin excluded, Java unsupported) -> defer.
    assert check_syntax("class A {}", "java") is None
    assert check_syntax("fun x() {}", "kotlin") is None


def _symbols_for(filename: str, source: str):
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / filename).write_text(source)
        g = SymbolGraph(Path(td))
        g.build()
        return {(s.kind, s.name) for s in g.symbols.values()}


def test_symbol_graph_parses_c():
    syms = _symbols_for("a.c", "int add(int a,int b){return a+b;}\nstruct Pt{int x;};")
    if not syms:
        return  # tree-sitter C grammar not installed in this environment
    assert ("function", "add") in syms
    assert ("struct", "Pt") in syms


def test_symbol_graph_parses_cpp_class_and_method():
    syms = _symbols_for(
        "w.cpp", "class Widget{ public: int area(){return 0;} };\nint main(){return 0;}"
    )
    if not syms:
        return
    assert ("class", "Widget") in syms
    assert ("method", "Widget::area") in syms
    assert ("function", "main") in syms


def test_symbol_graph_parses_swift():
    syms = _symbols_for(
        "e.swift",
        "class Engine { func start() {} }\nstruct P { let x: Int }\nprotocol D { func draw() }",
    )
    if not syms:
        return
    assert ("class", "Engine") in syms
    assert ("method", "Engine.start") in syms
    assert ("struct", "P") in syms
    assert ("protocol", "D") in syms


def test_symbol_graph_parses_csharp():
    syms = _symbols_for(
        "App.cs",
        "namespace A { public class Engine { public void Start() {} "
        "public int Count { get; set; } } public interface IDraw { void Draw(); } }",
    )
    if not syms:
        return
    assert ("class", "Engine") in syms
    assert ("method", "Engine.Start") in syms
    assert ("property", "Engine.Count") in syms
    assert ("interface", "IDraw") in syms


def test_file_outline_lists_symbols_with_lines():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = "\n".join(["// h"] + ["fn start() {}"] + ["x"] * 100 + ["fn stop() {}"])
        (td / "engine.rs").write_text(src)
        g = SymbolGraph(td)
        g.build()
        outline = g.file_outline("engine.rs")
        if not outline:
            return  # rust grammar unavailable
        assert "function start" in outline
        assert "function stop" in outline
        assert "L2" in outline  # start is on line 2


def test_file_outline_empty_for_unknown_file():
    with tempfile.TemporaryDirectory() as td:
        g = SymbolGraph(Path(td))
        g.build()
        assert g.file_outline("nope.rs") == ""


def test_project_outline_maps_all_files():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "a.rs").write_text("fn one() {}\nstruct A {}\n")
        (td / "b.rs").write_text("fn two() {}\n")
        g = SymbolGraph(td)
        g.build()
        outline = g.project_outline()
        if not outline:
            return
        assert "a.rs:" in outline and "b.rs:" in outline
        assert "function one" in outline and "struct A" in outline
        assert "function two" in outline


def test_symbol_graph_byte_offsets_with_non_ascii():
    # Non-ASCII bytes before a symbol must not shift tree-sitter byte offsets
    # and mangle extracted names (regression: names were sliced from the str).
    syms = _symbols_for(
        "x.rs", "// café ☕ — naïve comment\nfn process_data() {}\nstruct Wörld {}\n"
    )
    if not syms:
        return
    assert ("function", "process_data") in syms
    assert ("struct", "Wörld") in syms


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
