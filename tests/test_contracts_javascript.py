from misterdev.core.context.contracts.javascript_tree_sitter import (
    _extract_javascript_symbols_ts,
)


def _by_name(syms):
    return {s["name"]: s for s in syms}


def test_export_function():
    syms = _extract_javascript_symbols_ts("export function run(x) { return x }\n")
    if not syms:
        return  # javascript grammar unavailable
    by = _by_name(syms)
    assert "run" in by
    assert by["run"]["kind"] == "export function"
    assert "run(x)" in by["run"]["signature"]
    assert "return x" not in by["run"]["signature"]  # body dropped


def test_export_class_with_method():
    content = 'export class Service { query() { return "" } }\n'
    syms = _extract_javascript_symbols_ts(content)
    if not syms:
        return
    by = _by_name(syms)
    assert "Service" in by
    assert by["Service"]["kind"] == "export class"
    assert "Service.query" in by  # method nested under class
    assert by["Service.query"]["kind"] == "method"


def test_export_const():
    syms = _extract_javascript_symbols_ts("export const LIMIT = 10\n")
    if not syms:
        return
    by = _by_name(syms)
    assert "LIMIT" in by
    assert by["LIMIT"]["kind"] == "export const"
    assert "LIMIT = 10" in by["LIMIT"]["signature"]


def test_export_let_multiple_declarators():
    syms = _extract_javascript_symbols_ts("export let a = 1, b = 2\n")
    if not syms:
        return
    by = _by_name(syms)
    assert "a" in by and "b" in by
    assert by["a"]["kind"] == "export let"


def test_export_default_function():
    syms = _extract_javascript_symbols_ts("export default function App() {}\n")
    if not syms:
        return
    by = _by_name(syms)
    assert "App" in by
    assert by["App"]["kind"] == "export default function"


def test_export_default_class():
    syms = _extract_javascript_symbols_ts("export default class Foo {}\n")
    if not syms:
        return
    by = _by_name(syms)
    assert "Foo" in by
    assert by["Foo"]["kind"] == "export default class"


def test_non_exported_function_skipped():
    content = "export function shown() {}\nfunction hidden() {}\n"
    syms = _extract_javascript_symbols_ts(content)
    if not syms:
        return
    names = {s["name"] for s in syms}
    assert "shown" in names
    assert "hidden" not in names


def test_empty_and_garbage_input():
    assert _extract_javascript_symbols_ts("") == []
    assert _extract_javascript_symbols_ts("@@@ )))(((  <<< not js") == []
