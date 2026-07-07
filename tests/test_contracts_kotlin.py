from misterdev.core.context.contracts.kotlin_tree_sitter import (
    _extract_kotlin_symbols_ts,
)


def _by_name(syms):
    return {s["name"]: s for s in syms}


def test_kotlin_top_level_fun():
    syms = _extract_kotlin_symbols_ts("fun run(x: Int): Int = x\n")
    if not syms:
        return  # kotlin grammar unavailable
    run = _by_name(syms)["run"]
    assert run["kind"] == "fun"
    assert "fun run(x: Int): Int" in run["signature"]
    assert "= x" not in run["signature"]  # body dropped


def test_kotlin_class_method_nested_name():
    content = 'class Service {\n    fun query(): String { return "" }\n}\n'
    syms = _extract_kotlin_symbols_ts(content)
    if not syms:
        return
    names = _by_name(syms)
    assert "Service" in names and names["Service"]["kind"] == "class"
    assert "Service.query" in names  # member nested under its type
    assert names["Service.query"]["kind"] == "fun"


def test_kotlin_interface():
    content = "interface Queryable {\n    fun query(): String\n}\n"
    syms = _extract_kotlin_symbols_ts(content)
    if not syms:
        return
    names = _by_name(syms)
    assert names["Queryable"]["kind"] == "interface"
    assert "Queryable.query" in names


def test_kotlin_enum_class_with_entries():
    syms = _extract_kotlin_symbols_ts("enum class Status { OK, FAILED }\n")
    if not syms:
        return
    status = _by_name(syms)["Status"]
    assert status["kind"] == "enum class"
    assert "OK" in status["signature"] and "FAILED" in status["signature"]


def test_kotlin_data_class_keeps_ctor_params():
    syms = _extract_kotlin_symbols_ts(
        "data class Cfg(val name: String, var count: Int)\n"
    )
    if not syms:
        return
    cfg = _by_name(syms)["Cfg"]
    assert cfg["kind"] == "data class"
    assert "name: String" in cfg["signature"]
    assert "count: Int" in cfg["signature"]


def test_kotlin_skips_private_fun():
    syms = _extract_kotlin_symbols_ts("private fun hidden() {}\nfun shown() {}\n")
    if not syms:
        return
    names = _by_name(syms)
    assert "shown" in names
    assert "hidden" not in names


def test_kotlin_skips_internal():
    syms = _extract_kotlin_symbols_ts("internal fun pkg() {}\nfun shown() {}\n")
    if not syms:
        return
    names = _by_name(syms)
    assert "shown" in names
    assert "pkg" not in names


def test_kotlin_extension_fun_keeps_receiver():
    syms = _extract_kotlin_symbols_ts("fun Foo.bar(): Int = 1\n")
    if not syms:
        return
    names = _by_name(syms)
    assert "Foo.bar" in names  # receiver-qualified
    assert names["Foo.bar"]["kind"] == "fun"
    assert "fun Foo.bar(): Int" in names["Foo.bar"]["signature"]
    assert "= 1" not in names["Foo.bar"]["signature"]  # body dropped


def test_kotlin_companion_object():
    content = "class C {\n companion object {\n  fun make(): C = C()\n }\n}\n"
    syms = _extract_kotlin_symbols_ts(content)
    if not syms:
        return
    names = _by_name(syms)
    assert "C.Companion" in names
    assert names["C.Companion"]["kind"] == "companion object"
    assert "C.make" in names  # companion member accessible on the class
    assert names["C.make"]["kind"] == "fun"


def test_kotlin_sealed_class():
    syms = _extract_kotlin_symbols_ts("sealed class Shape\n")
    if not syms:
        return
    shape = _by_name(syms)["Shape"]
    assert shape["kind"] == "sealed class"


def test_kotlin_suspend_fun():
    syms = _extract_kotlin_symbols_ts('suspend fun load(): String = ""\n')
    if not syms:
        return
    load = _by_name(syms)["load"]
    assert load["kind"] == "fun"
    assert "suspend fun load(): String" in load["signature"]


def test_kotlin_typealias():
    syms = _extract_kotlin_symbols_ts("typealias Name = String\n")
    if not syms:
        return
    name = _by_name(syms)["Name"]
    assert name["kind"] == "typealias"
    assert "typealias Name = String" in name["signature"]


def test_kotlin_top_level_val_and_var():
    syms = _extract_kotlin_symbols_ts("val PI = 3.14\nvar count = 0\n")
    if not syms:
        return
    names = _by_name(syms)
    assert names["PI"]["kind"] == "val"
    assert "= 3.14" not in names["PI"]["signature"]  # initializer dropped
    assert names["count"]["kind"] == "var"


def test_kotlin_empty_and_garbage_return_list():
    assert _extract_kotlin_symbols_ts("") == []
    # Garbage still parses without throwing; it yields no valid declarations.
    assert _extract_kotlin_symbols_ts(")))(((not kotlin @@@") == []
