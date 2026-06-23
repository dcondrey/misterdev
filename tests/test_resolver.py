from pathlib import Path

from my_project_orchestrator.core.resolver import ErrorResolver, ErrorLocation
from my_project_orchestrator.core.topography import SymbolGraph, SymbolNode


def _make_graph():
    graph = SymbolGraph.__new__(SymbolGraph)
    graph.project_path = Path("/fake/project")
    graph.symbols = {}
    graph.parser = None
    return graph


def _make_resolver(graph=None):
    graph = graph or _make_graph()
    return ErrorResolver(Path("/fake/project"), graph)


def test_parse_python_traceback():
    r = _make_resolver()
    locs = r.resolve_errors(
        '  File "src/main.py", line 42, in do_stuff\nValueError: bad'
    )
    assert len(locs) == 1
    assert locs[0].file_path == "src/main.py"
    assert locs[0].line == 42


def test_parse_generic_file_line():
    r = _make_resolver()
    locs = r.resolve_errors("src/lib.rs:17: mismatched types")
    assert len(locs) == 1
    assert locs[0].file_path == "src/lib.rs"
    assert locs[0].line == 17
    assert "mismatched" in locs[0].message


def test_parse_pytest_error():
    r = _make_resolver()
    locs = r.resolve_errors("tests/test_foo.py:5: AssertionError")
    assert len(locs) == 1
    assert locs[0].file_path == "tests/test_foo.py"


def test_dedup_same_file_line():
    r = _make_resolver()
    error_output = (
        "src/lib.rs:10: error one\nsrc/lib.rs:10: error two\nsrc/lib.rs:20: error three"
    )
    locs = r.resolve_errors(error_output)
    assert len(locs) == 2
    lines = {loc.line for loc in locs}
    assert lines == {10, 20}


def test_non_source_file_ignored():
    r = _make_resolver()
    locs = r.resolve_errors("README.md:5: something")
    assert len(locs) == 0


def test_no_errors_returns_empty():
    r = _make_resolver()
    locs = r.resolve_errors("Build succeeded!\nAll tests pass.")
    assert locs == []


def test_absolute_path_relativized():
    r = _make_resolver()
    locs = r.resolve_errors('  File "/fake/project/src/main.py", line 10, in foo')
    assert len(locs) == 1
    assert locs[0].file_path == "src/main.py"


def test_absolute_path_outside_project_ignored():
    r = _make_resolver()
    locs = r.resolve_errors('  File "/other/path/main.py", line 10, in foo')
    assert locs == []


def test_symbol_attribution():
    graph = _make_graph()
    node = SymbolNode(
        "do_stuff", "src/main.py", "function", 40, 50, "def do_stuff(): pass"
    )
    graph.symbols["src/main.py:do_stuff"] = node
    r = _make_resolver(graph)
    locs = r.resolve_errors("src/main.py:42: error here")
    assert len(locs) == 1
    assert locs[0].symbol == "do_stuff"


def test_format_for_llm_empty():
    r = _make_resolver()
    assert "No specific error" in r.format_for_llm([])


def test_format_for_llm_with_locations():
    r = _make_resolver()
    locs = [ErrorLocation("src/lib.rs", 10, "type mismatch", "validate")]
    output = r.format_for_llm(locs)
    assert "validate" in output
    assert "src/lib.rs:10" in output
    assert "type mismatch" in output


def test_error_location_repr():
    loc = ErrorLocation("src/lib.rs", 10, "bad type", "validate")
    s = repr(loc)
    assert "src/lib.rs:10" in s
    assert "validate" in s


def test_looks_like_source():
    r = _make_resolver()
    assert r._looks_like_source("src/main.py")
    assert r._looks_like_source("lib.rs")
    assert r._looks_like_source("app.ts")
    assert not r._looks_like_source("README.md")
    assert not r._looks_like_source("data.csv")
