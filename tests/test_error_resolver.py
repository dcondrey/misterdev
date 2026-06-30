import tempfile
from pathlib import Path

from my_project_orchestrator.core.execution.error_resolver import (
    ErrorResolver,
    ErrorLocation,
)
from my_project_orchestrator.core.context.topography import SymbolGraph, SymbolNode


def _graph(*nodes):
    # Real SymbolGraph (bypassing the parser) so the resolver exercises the
    # graph's symbol_at_line/callers_of, where the attribution logic now lives.
    g = SymbolGraph.__new__(SymbolGraph)
    g.symbols = {f"{n.file_path}:{n.name}": n for n in nodes}
    return g


def test_resolve_python_traceback_location():
    r = ErrorResolver(Path("."))
    out = 'File "src/app.py", line 42\n    raise ValueError'
    locs = r.resolve_errors(out)
    assert any(loc.file == "src/app.py" and loc.line == 42 for loc in locs)


def test_resolve_colon_location_and_rust_arrow():
    r = ErrorResolver(Path("."))
    out = "src/lib.rs:10: error: bad\n --> src/main.rs:7:3"
    files = {(loc.file, loc.line) for loc in r.resolve_errors(out)}
    assert ("src/lib.rs", 10) in files
    assert ("src/main.rs", 7) in files


def test_resolve_dedups_repeated_locations():
    r = ErrorResolver(Path("."))
    out = "a.py:5: error one\na.py:5: error two"
    locs = r.resolve_errors(out)
    assert len([loc for loc in locs if loc.file == "a.py" and loc.line == 5]) == 1


def test_resolve_no_locations_returns_empty():
    assert (
        ErrorResolver(Path(".")).resolve_errors("a vague failure, no file refs") == []
    )


def test_read_snippet_marks_error_line():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "m.py").write_text("\n".join(f"line{i}" for i in range(1, 21)))
        r = ErrorResolver(td)
        locs = r.resolve_errors("m.py:10: error: boom")
        loc = next(loc for loc in locs if loc.file == "m.py")
        assert "line10" in loc.snippet
        assert ">" in loc.snippet  # the error line is marked


def test_format_for_llm_includes_locations_and_caps():
    r = ErrorResolver(Path("."))
    locs = [ErrorLocation(f"f{i}.py", i, snippet=f"code{i}") for i in range(15)]
    out = r.format_for_llm(locs)
    assert "## Error Attribution" in out
    assert "f0.py:0" in out
    assert "code0" in out
    # capped at 10 entries
    assert "f12.py" not in out


def test_format_for_llm_empty():
    assert ErrorResolver(Path(".")).format_for_llm([]) == ""


def test_no_symbol_attribution_without_graph():
    # With no graph, locations carry no symbol (prior behavior preserved).
    locs = ErrorResolver(Path(".")).resolve_errors("src/app.py:5: boom")
    assert locs and all(loc.symbol is None for loc in locs)


def test_symbol_attribution_from_graph():
    # Symbol bounds are 0-indexed tree-sitter rows; a 1-indexed error line maps
    # to row line-1. A symbol on rows 40-50 covers error lines 41-51.
    graph = _graph(SymbolNode("do_stuff", "src/main.py", "function", 40, 50, "x"))
    r = ErrorResolver(Path("."), graph)
    locs = r.resolve_errors("src/main.py:42: error here")
    assert len(locs) == 1
    assert locs[0].symbol == "do_stuff"


def test_symbol_attribution_index_boundary():
    # Single-line symbol on row 9 == error line 10; line 11 (row 10) is outside.
    graph = _graph(SymbolNode("f", "a.py", "function", 9, 9, "x"))
    r = ErrorResolver(Path("."), graph)
    assert r.resolve_errors("a.py:10: e")[0].symbol == "f"
    assert r.resolve_errors("a.py:11: e")[0].symbol is None


def test_symbol_attribution_picks_narrowest_enclosing():
    # A method nested in a class: the narrower enclosing symbol wins.
    graph = _graph(
        SymbolNode("Cls", "a.py", "class", 0, 100, "x"),
        SymbolNode("m", "a.py", "method", 40, 50, "x"),
    )
    r = ErrorResolver(Path("."), graph)
    assert r.resolve_errors("a.py:45: e")[0].symbol == "m"


def test_format_for_llm_includes_symbol_and_callers():
    callee = SymbolNode("validate", "a.py", "function", 9, 9, "x")
    callee.incoming_calls.add("a.py:run")
    graph = _graph(callee, SymbolNode("run", "a.py", "function", 20, 30, "x"))
    r = ErrorResolver(Path("."), graph)
    out = r.format_for_llm(r.resolve_errors("a.py:10: type error"))
    assert "Symbol: `validate`" in out
    assert "Called by: run" in out


def test_callers_not_conflated_across_same_named_symbols():
    # Two files each define `run`; only a.py's caller must be reported for an
    # a.py error — matching by unique key, not bare name.
    a_run = SymbolNode("run", "a.py", "function", 9, 9, "x")
    a_run.incoming_calls.add("a.py:a_caller")
    b_run = SymbolNode("run", "b.py", "function", 9, 9, "x")
    b_run.incoming_calls.add("b.py:b_caller")
    graph = _graph(
        a_run,
        b_run,
        SymbolNode("a_caller", "a.py", "function", 20, 30, "x"),
        SymbolNode("b_caller", "b.py", "function", 20, 30, "x"),
    )
    r = ErrorResolver(Path("."), graph)
    out = r.format_for_llm(r.resolve_errors("a.py:10: boom"))
    assert "Called by: a_caller" in out
    assert "b_caller" not in out


def test_symbol_attribution_subtarget_relative_path():
    # An error path reported relative to a sub-target's cwd (src/app.py) still
    # resolves against the root-relative graph key via the unique-suffix match.
    graph = _graph(SymbolNode("handler", "frontend/src/app.py", "function", 5, 15, "x"))
    r = ErrorResolver(Path("."), graph)
    assert r.resolve_errors("src/app.py:8: e")[0].symbol == "handler"


def test_symbol_attribution_ambiguous_suffix_not_guessed():
    # Same basename under two targets: an ambiguous suffix must NOT be guessed.
    graph = _graph(
        SymbolNode("f", "a/x.py", "function", 5, 15, "x"),
        SymbolNode("g", "b/x.py", "function", 5, 15, "x"),
    )
    r = ErrorResolver(Path("."), graph)
    assert r.resolve_errors("x.py:8: e")[0].symbol is None
