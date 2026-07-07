from misterdev.core.context.contracts.c_tree_sitter import _extract_c_symbols_ts


def _by_name(syms):
    return {s["name"]: s for s in syms}


def test_c_function_definition():
    syms = _extract_c_symbols_ts("int run(int x) { return x; }\n")
    if not syms:
        return  # c grammar unavailable
    sym = _by_name(syms)["run"]
    assert sym["kind"] == "fn"
    assert "int run(int x)" in sym["signature"]
    assert "return" not in sym["signature"]  # body dropped


def test_c_function_prototype():
    syms = _extract_c_symbols_ts("void init(void);\n")
    if not syms:
        return
    sym = _by_name(syms)["init"]
    assert sym["kind"] == "fn"
    assert "void init(void)" in sym["signature"]
    assert ";" not in sym["signature"]


def test_c_struct_with_fields():
    syms = _extract_c_symbols_ts("struct Cfg { char *name; int n; };\n")
    if not syms:
        return
    sym = _by_name(syms)["Cfg"]
    assert sym["kind"] == "struct"
    assert "char *name" in sym["signature"]
    assert "int n" in sym["signature"]


def test_c_enum_with_enumerators():
    syms = _extract_c_symbols_ts("enum Status { OK, FAILED };\n")
    if not syms:
        return
    sym = _by_name(syms)["Status"]
    assert sym["kind"] == "enum"
    assert "OK" in sym["signature"]
    assert "FAILED" in sym["signature"]


def test_c_typedef():
    syms = _extract_c_symbols_ts("typedef struct Cfg Cfg;\n")
    if not syms:
        return
    names = {s["name"]: s for s in syms if s["kind"] == "typedef"}
    assert "Cfg" in names
    assert "typedef struct Cfg Cfg" in names["Cfg"]["signature"]


def test_c_skips_static_function():
    syms = _extract_c_symbols_ts(
        "int visible(void) { return 1; }\nstatic int hidden(void) { return 0; }\n"
    )
    if not syms:
        return
    names = _by_name(syms)
    assert "visible" in names
    assert "hidden" not in names


def test_c_empty_and_garbage():
    assert _extract_c_symbols_ts("") == []
    # Garbage never yields public C symbols (parser tolerates or returns []).
    assert _extract_c_symbols_ts("@@@ ??? not c at all %%%") == []
