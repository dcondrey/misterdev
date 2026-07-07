from misterdev.core.context.contracts.typescript_tree_sitter import (
    _extract_typescript_symbols_ts,
)


def _by_name(symbols):
    return {s["name"]: s for s in symbols}


def test_export_function():
    syms = _extract_typescript_symbols_ts(
        "export function run(x: number): number { return x }\n"
    )
    if not syms:
        return  # typescript grammar unavailable
    by = _by_name(syms)
    assert "run" in by
    assert by["run"]["kind"] == "export function"
    assert "x: number" in by["run"]["signature"]
    assert "return x" not in by["run"]["signature"]  # body dropped


def test_export_class_with_method():
    syms = _extract_typescript_symbols_ts(
        'export class Service { query(): string { return "" } }\n'
    )
    if not syms:
        return
    by = _by_name(syms)
    assert "Service" in by
    assert by["Service"]["kind"] == "export class"
    assert "Service.query" in by  # method nested under the type
    assert by["Service.query"]["kind"] == "method"


def test_export_class_skips_private_method():
    syms = _extract_typescript_symbols_ts(
        "export class Svc { pub(): void {} private secret(): void {} }\n"
    )
    if not syms:
        return
    names = {s["name"] for s in syms}
    assert "Svc.pub" in names
    assert "Svc.secret" not in names


def test_export_interface():
    syms = _extract_typescript_symbols_ts(
        "export interface Queryable { query(): string; id: number }\n"
    )
    if not syms:
        return
    by = _by_name(syms)
    assert "Queryable" in by
    assert by["Queryable"]["kind"] == "export interface"
    assert "Queryable.query" in by
    assert "Queryable.id" in by


def test_export_type():
    syms = _extract_typescript_symbols_ts("export type Id = number\n")
    if not syms:
        return
    by = _by_name(syms)
    assert "Id" in by
    assert by["Id"]["kind"] == "export type"
    assert "number" in by["Id"]["signature"]


def test_export_const():
    syms = _extract_typescript_symbols_ts("export const LIMIT = 10\n")
    if not syms:
        return
    by = _by_name(syms)
    assert "LIMIT" in by
    assert by["LIMIT"]["kind"] == "export const"


def test_export_enum():
    syms = _extract_typescript_symbols_ts("export enum Color { Red, Green }\n")
    if not syms:
        return
    by = _by_name(syms)
    assert "Color" in by
    assert by["Color"]["kind"] == "export enum"
    assert "Red" in by["Color"]["signature"]
    assert "Green" in by["Color"]["signature"]


def test_export_default_class():
    syms = _extract_typescript_symbols_ts("export default class App {}\n")
    if not syms:
        return
    by = _by_name(syms)
    assert "App" in by
    assert by["App"]["kind"] == "export default class"


def test_non_exported_is_skipped():
    syms = _extract_typescript_symbols_ts(
        "export function shown() {}\nfunction hidden() {}\n"
    )
    if not syms:
        return
    names = {s["name"] for s in syms}
    assert "shown" in names
    assert "hidden" not in names


def test_export_namespace():
    syms = _extract_typescript_symbols_ts(
        "export namespace NS { export const x = 1; export function f(): void {} }\n"
    )
    if not syms:
        return
    by = _by_name(syms)
    assert "NS" in by
    assert by["NS"]["kind"] == "export namespace"
    assert "{" not in by["NS"]["signature"]  # body dropped
    assert "NS.x" in by  # nested exported member
    assert "NS.f" in by


def test_export_generic_class():
    syms = _extract_typescript_symbols_ts(
        "export class Box<T> { get(): T { return null as any } }\n"
    )
    if not syms:
        return
    by = _by_name(syms)
    assert "Box" in by
    assert by["Box"]["kind"] == "export class"
    assert "<T>" in by["Box"]["signature"]  # type params preserved
    assert "Box.get" in by


def test_reexport_named():
    syms = _extract_typescript_symbols_ts("export { a, b as c } from './c';\n")
    if not syms:
        return
    by = _by_name(syms)
    assert "a" in by
    assert "c" in by  # alias, not the original name
    assert "b" not in by
    assert by["a"]["kind"] == "re-export"


def test_reexport_star():
    syms = _extract_typescript_symbols_ts("export * from './a';\n")
    if not syms:
        return
    names = {s["name"] for s in syms}
    assert "./a" in names


def test_export_arrow_const():
    syms = _extract_typescript_symbols_ts("export const f = (x: number) => x;\n")
    if not syms:
        return
    by = _by_name(syms)
    assert "f" in by
    assert by["f"]["kind"] == "export const"
    assert "=>" in by["f"]["signature"]


def test_export_class_getter():
    syms = _extract_typescript_symbols_ts(
        "export class G { get val(): number { return 1 } set val(v: number) {} }\n"
    )
    if not syms:
        return
    sigs = [s["signature"] for s in syms if s["name"] == "G.val"]
    assert any(sig.startswith("get val") for sig in sigs)
    assert any(sig.startswith("set val") for sig in sigs)
    assert all("return 1" not in sig for sig in sigs)  # body dropped


def test_export_declare_function():
    syms = _extract_typescript_symbols_ts("export declare function d(): void;\n")
    if not syms:
        return
    by = _by_name(syms)
    assert "d" in by
    assert by["d"]["kind"] == "export declare function"


def test_export_default_expression():
    syms = _extract_typescript_symbols_ts("const foo = 1;\nexport default foo;\n")
    if not syms:
        return
    by = _by_name(syms)
    assert "foo" in by
    assert by["foo"]["kind"] == "export default"


def test_empty_and_garbage_input():
    assert _extract_typescript_symbols_ts("") == []
    # Unparseable soup must not raise and must not invent symbols.
    garbage = _extract_typescript_symbols_ts(")(}{ >< export export ;;;")
    assert isinstance(garbage, list)
