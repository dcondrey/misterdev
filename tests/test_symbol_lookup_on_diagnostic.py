"""T2.2 — a diagnostic that NAMES a symbol surfaces that symbol's definition.

On "cannot find X" / "expected `A`, found `B`" / tsc "Cannot find name 'X'", the
resolver must look X up in the symbol graph and surface X's DEFINITION (where it is
declared) into the repair context — not merely attribute the error line to its
enclosing symbol. Only names that resolve to a real project symbol are surfaced, so
compiler primitives (`u32`, `string`) never add noise.
"""

from pathlib import Path

from misterdev.core.context.topography import SymbolGraph, SymbolNode
from misterdev.core.execution.error_resolver import ErrorResolver


def _graph(*nodes):
    g = SymbolGraph.__new__(SymbolGraph)
    g.symbols = {f"{n.file_path}:{n.name}": n for n in nodes}
    return g


def test_rust_cannot_find_function_surfaces_definition():
    graph = _graph(
        SymbolNode("helper", "src/util.rs", "function", 11, 15, "fn helper() {}")
    )
    r = ErrorResolver(Path("."), graph)
    out = r.format_for_llm(
        [], error_output="error[E0425]: cannot find function `helper` in this scope"
    )
    assert "Referenced symbol definitions" in out
    assert "`helper`" in out
    assert "src/util.rs:12" in out  # 0-indexed row 11 -> 1-indexed line 12
    assert "fn helper" in out  # the definition body is included


def test_tsc_cannot_find_name_surfaces_definition():
    graph = _graph(
        SymbolNode("Widget", "src/widget.ts", "class", 4, 30, "class Widget {}")
    )
    r = ErrorResolver(Path("."), graph)
    out = r.format_for_llm(
        [], error_output="src/a.ts(3,5): error TS2304: Cannot find name 'Widget'."
    )
    assert "`Widget`" in out
    assert "src/widget.ts:5" in out


def test_type_mismatch_surfaces_named_type_definition():
    graph = _graph(
        SymbolNode("Celsius", "src/units.rs", "struct", 9, 12, "struct Celsius(f64);")
    )
    r = ErrorResolver(Path("."), graph)
    out = r.format_for_llm([], error_output="expected `Celsius`, found `Fahrenheit`")
    assert "`Celsius`" in out
    assert "src/units.rs:10" in out


def test_unknown_or_primitive_name_is_not_surfaced():
    graph = _graph(SymbolNode("helper", "src/util.rs", "function", 11, 15, "x"))
    r = ErrorResolver(Path("."), graph)
    # `u32` is not a project symbol -> no referenced-definition section at all.
    out = r.format_for_llm([], error_output="expected `u32`, found `&str`")
    assert "Referenced symbol definitions" not in out


def test_no_graph_no_referenced_section():
    r = ErrorResolver(Path("."))  # no graph
    out = r.format_for_llm(
        [], error_output="cannot find function `helper` in this scope"
    )
    assert "Referenced symbol definitions" not in out


def test_error_output_omitted_preserves_prior_behavior():
    # Backward compatibility: calling without error_output yields only attribution.
    graph = _graph(SymbolNode("helper", "src/util.rs", "function", 11, 15, "x"))
    r = ErrorResolver(Path("."), graph)
    out = r.format_for_llm([])
    assert out == ""
