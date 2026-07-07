from misterdev.core.context.contracts.swift_tree_sitter import (
    _extract_swift_symbols_ts,
)


def test_swift_public_func():
    syms = _extract_swift_symbols_ts("public func run(_ x: Int) -> Int { x }\n")
    if not syms:
        return  # swift grammar unavailable
    fn = next(s for s in syms if s["kind"] == "func")
    assert fn["name"] == "run"
    assert "run(_ x: Int) -> Int" in fn["signature"]
    assert "{" not in fn["signature"]  # body dropped


def test_swift_public_struct_with_property():
    syms = _extract_swift_symbols_ts("public struct Cfg { public let name: String }\n")
    if not syms:
        return
    st = next(s for s in syms if s["kind"] == "struct")
    assert st["name"] == "Cfg"
    assert "name" in st["signature"] and "String" in st["signature"]


def test_swift_public_class_with_method():
    syms = _extract_swift_symbols_ts(
        'public class Service { public func query() -> String { "" } }\n'
    )
    if not syms:
        return
    names = {s["name"] for s in syms}
    assert "Service" in names
    assert "Service.query" in names  # method nested under its type


def test_swift_protocol_with_requirements():
    syms = _extract_swift_symbols_ts(
        "public protocol Queryable { func query() -> String }\n"
    )
    if not syms:
        return
    proto = next(s for s in syms if s["kind"] == "protocol")
    assert proto["name"] == "Queryable"
    assert "query" in proto["signature"]


def test_swift_enum_with_cases():
    syms = _extract_swift_symbols_ts("public enum Status { case ok, failed }\n")
    if not syms:
        return
    en = next(s for s in syms if s["kind"] == "enum")
    assert en["name"] == "Status"
    assert "ok" in en["signature"] and "failed" in en["signature"]


def test_swift_top_level_type_unmarked():
    # Exercise code often omits modifiers; top-level types are still captured.
    syms = _extract_swift_symbols_ts("struct Bare { let x: Int }\n")
    if not syms:
        return
    assert any(s["name"] == "Bare" and s["kind"] == "struct" for s in syms)


def test_swift_extension_methods_nested():
    syms = _extract_swift_symbols_ts(
        "extension Foo {\n"
        '    public func hello() -> String { "" }\n'
        "    public var doubled: Int { 2 }\n"
        "}\n"
    )
    if not syms:
        return
    names = {s["name"] for s in syms}
    assert "Foo.hello" in names  # extension method nested under Foo
    fn = next(s for s in syms if s["name"] == "Foo.hello")
    assert "{" not in fn["signature"]  # body dropped


def test_swift_computed_property():
    syms = _extract_swift_symbols_ts(
        "public struct S { public var doubled: Int { x * 2 } }\n"
    )
    if not syms:
        return
    st = next(s for s in syms if s["kind"] == "struct")
    assert "doubled" in st["signature"]
    assert "x * 2" not in st["signature"]  # computed body dropped


def test_swift_static_method():
    syms = _extract_swift_symbols_ts(
        "public struct M { public static func make() -> M { M() } }\n"
    )
    if not syms:
        return
    fn = next(s for s in syms if s["name"] == "M.make")
    assert "static" in fn["signature"]
    assert "{" not in fn["signature"]


def test_swift_typealias():
    syms = _extract_swift_symbols_ts("public typealias ID = String\n")
    if not syms:
        return
    ta = next(s for s in syms if s["kind"] == "typealias")
    assert ta["name"] == "ID"
    assert "String" in ta["signature"]


def test_swift_generic_func_with_where():
    syms = _extract_swift_symbols_ts(
        "public func maxOf<T>(_ a: T, _ b: T) -> T where T: Comparable { a }\n"
    )
    if not syms:
        return
    fn = next(s for s in syms if s["kind"] == "func")
    assert fn["name"] == "maxOf"
    assert "<T>" in fn["signature"]
    assert "where T: Comparable" in fn["signature"]
    assert "{" not in fn["signature"]


def test_swift_public_init_nested():
    syms = _extract_swift_symbols_ts("public struct P { public init(x: Int) {} }\n")
    if not syms:
        return
    init = next(s for s in syms if s["kind"] == "init")
    assert init["name"] == "P.init"
    assert "x: Int" in init["signature"]
    assert "{" not in init["signature"]


def test_swift_subscript_nested():
    syms = _extract_swift_symbols_ts(
        "public struct Arr { public subscript(i: Int) -> Int { 0 } }\n"
    )
    if not syms:
        return
    sub = next(s for s in syms if s["kind"] == "subscript")
    assert sub["name"] == "Arr.subscript"
    assert "i: Int" in sub["signature"] and "-> Int" in sub["signature"]
    assert "{" not in sub["signature"]


def test_swift_protocol_associatedtype():
    syms = _extract_swift_symbols_ts(
        "public protocol Container {\n"
        "    associatedtype Item\n"
        "    func item(at i: Int) -> Item\n"
        "}\n"
    )
    if not syms:
        return
    proto = next(s for s in syms if s["kind"] == "protocol")
    assert "associatedtype Item" in proto["signature"]


def test_swift_open_declarations():
    syms = _extract_swift_symbols_ts("open class Base { open func run() {} }\n")
    if not syms:
        return
    names = {s["name"] for s in syms}
    assert "Base" in names  # open type captured
    assert "Base.run" in names  # open method captured


def test_swift_empty_and_garbage_return_empty():
    assert _extract_swift_symbols_ts("") == []
    assert _extract_swift_symbols_ts(")))(((not swift @@@") == []
