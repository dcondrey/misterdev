"""H4 — SymbolGraph's file-scoped queries use a memoized per-file index.

file_symbols / _match_files / symbol_at_line were each a full O(all-symbols) scan.
A memoized `(file_path -> [(key, node)])` index makes them file-local and is rebuilt
only when the symbol set is replaced/resized. Behavior is unchanged; this locks it.
"""

from misterdev.core.context.topography import SymbolGraph, SymbolNode


def _graph(*nodes):
    g = SymbolGraph.__new__(SymbolGraph)
    g.symbols = {f"{n.file_path}:{n.name}": n for n in nodes}
    return g


def test_file_index_groups_by_file_sorted_by_start_line():
    g = _graph(
        SymbolNode("b", "a.py", "function", 20, 25, "x"),
        SymbolNode("a", "a.py", "function", 5, 10, "x"),
        SymbolNode("z", "other.py", "function", 0, 3, "x"),
    )
    idx = g._file_index()
    assert set(idx) == {"a.py", "other.py"}
    assert [node.name for _k, node in idx["a.py"]] == ["a", "b"]  # sorted by start_line


def test_file_index_is_memoized_and_rebuilds_on_replacement():
    g = _graph(SymbolNode("a", "a.py", "function", 0, 5, "x"))
    first = g._file_index()
    assert g._file_index() is first  # same object while symbols unchanged
    g.symbols = {"b.py:b": SymbolNode("b", "b.py", "function", 0, 5, "x")}
    assert g._file_index() is not first  # rebuilt after the symbol set is replaced
    assert set(g._file_index()) == {"b.py"}


def test_public_methods_unchanged():
    g = _graph(
        SymbolNode("Cls", "a.py", "class", 0, 100, "x"),
        SymbolNode("m", "a.py", "method", 40, 50, "x"),
    )
    assert [s.name for s in g.file_symbols("a.py")] == ["Cls", "m"]
    # narrowest enclosing span wins (method over class)
    assert g.symbol_at_line("a.py", 45) == "a.py:m"
    assert g.symbol_at_line("a.py", 5) == "a.py:Cls"
    assert g.symbol_at_line("a.py", 200) is None
    assert g._match_files("a.py") == {"a.py"}


def test_unique_suffix_match_preserved():
    g = _graph(SymbolNode("h", "frontend/src/app.ts", "function", 5, 15, "x"))
    assert g._match_files("src/app.ts") == {"frontend/src/app.ts"}
    # ambiguous suffix -> no match
    g2 = _graph(
        SymbolNode("f", "a/x.py", "function", 5, 15, "x"),
        SymbolNode("g", "b/x.py", "function", 5, 15, "x"),
    )
    assert g2._match_files("x.py") == set()
