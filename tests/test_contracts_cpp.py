from misterdev.core.context.contracts.cpp_tree_sitter import _extract_cpp_symbols_ts


def _by_name(syms):
    return {s["name"]: s for s in syms}


def test_free_function():
    syms = _extract_cpp_symbols_ts("int run(int x) { return x; }\n")
    if not syms:
        return  # cpp grammar unavailable
    fn = _by_name(syms)["run"]
    assert fn["kind"] == "function"
    assert "int x" in fn["signature"]
    assert "return" not in fn["signature"]  # body dropped


def test_function_prototype():
    syms = _extract_cpp_symbols_ts("void ping(int n);\n")
    if not syms:
        return
    fn = _by_name(syms)["ping"]
    assert fn["kind"] == "function"
    assert "int n" in fn["signature"]


def test_class_public_method_and_private_excluded():
    content = (
        "class Service {\n"
        "public:\n"
        "  std::string query();\n"
        "private:\n"
        "  int secret;\n"
        "  void hidden();\n"
        "};\n"
    )
    syms = _extract_cpp_symbols_ts(content)
    if not syms:
        return
    names = _by_name(syms)
    assert "Service" in names and names["Service"]["kind"] == "class"
    assert "Service::query" in names
    assert names["Service::query"]["kind"] == "method"
    assert "Service::hidden" not in names  # private method excluded
    assert "secret" not in names["Service"]["signature"]  # private field excluded


def test_struct_public_by_default():
    content = "struct Cfg { std::string name; int compute(); };\n"
    syms = _extract_cpp_symbols_ts(content)
    if not syms:
        return
    names = _by_name(syms)
    assert names["Cfg"]["kind"] == "struct"
    assert "name" in names["Cfg"]["signature"]  # struct field is public
    assert "Cfg::compute" in names


def test_enum_with_enumerators():
    syms = _extract_cpp_symbols_ts("enum class Status { Ok, Failed };\n")
    if not syms:
        return
    e = _by_name(syms)["Status"]
    assert e["kind"] == "enum"
    assert "Ok" in e["signature"] and "Failed" in e["signature"]


def test_template_function():
    syms = _extract_cpp_symbols_ts(
        "template<typename T> T identity(T x) { return x; }\n"
    )
    if not syms:
        return
    fn = _by_name(syms)["identity"]
    assert fn["kind"] == "function"
    assert "template" in fn["signature"]


def test_template_class():
    syms = _extract_cpp_symbols_ts(
        "template<typename T> class Box { public: T get(); };\n"
    )
    if not syms:
        return
    names = _by_name(syms)
    assert names["Box"]["kind"] == "class"
    assert "template" in names["Box"]["signature"]
    assert "Box::get" in names


def test_typedef_and_using():
    syms = _extract_cpp_symbols_ts(
        "typedef unsigned int uint_t;\nusing Id = unsigned long;\n"
    )
    if not syms:
        return
    names = _by_name(syms)
    assert names["uint_t"]["kind"] == "typedef"
    assert names["Id"]["kind"] == "using"


def test_namespace_functions_are_walked():
    syms = _extract_cpp_symbols_ts("namespace app { void init(); int helper(); }\n")
    if not syms:
        return
    names = _by_name(syms)
    assert "init" in names and "helper" in names


def test_out_of_line_method_definition():
    syms = _extract_cpp_symbols_ts(
        "std::string Service::query() const { return x_; }\n"
    )
    if not syms:
        return
    fn = _by_name(syms)["Service::query"]
    assert "const" in fn["signature"]
    assert "return" not in fn["signature"]  # body dropped


def test_free_operator_overload():
    syms = _extract_cpp_symbols_ts("bool operator==(const A& a, const A& b);\n")
    if not syms:
        return
    op = _by_name(syms)["operator=="]
    assert op["kind"] == "function"
    assert "const A& a" in op["signature"]


def test_member_operator_returning_reference():
    # reference_declarator does not tag its inner declarator, so this used to
    # be swallowed as a field instead of surfacing as a method.
    content = "class Vec { public: Vec& operator+=(const Vec& o); };\n"
    syms = _extract_cpp_symbols_ts(content)
    if not syms:
        return
    names = _by_name(syms)
    assert "Vec::operator+=" in names
    assert names["Vec::operator+="]["kind"] == "method"
    assert "operator+=" not in names["Vec"]["signature"]  # not misfiled as a field


def test_method_returning_reference_is_captured():
    content = "class C { public: Bar& ref(); Foo* ptr(); };\n"
    syms = _extract_cpp_symbols_ts(content)
    if not syms:
        return
    names = _by_name(syms)
    assert names["C::ref"]["kind"] == "method"
    assert names["C::ptr"]["kind"] == "method"


def test_nested_class_and_struct():
    content = (
        "class Outer {\n"
        "public:\n"
        "  class Inner { public: int v; };\n"
        "  struct Item { int a; };\n"
        "};\n"
    )
    syms = _extract_cpp_symbols_ts(content)
    if not syms:
        return
    names = _by_name(syms)
    assert names["Outer::Inner"]["kind"] == "class"
    assert names["Outer::Item"]["kind"] == "struct"
    # nested types no longer leak into the outer record signature
    assert "Inner" not in names["Outer"]["signature"]


def test_nested_private_type_excluded():
    content = (
        "class Priv {\n  class Hidden { public: int x; };\npublic:\n  int shown;\n};\n"
    )
    syms = _extract_cpp_symbols_ts(content)
    if not syms:
        return
    names = _by_name(syms)
    assert "Priv::Hidden" not in names  # private nested type excluded
    assert "shown" in names["Priv"]["signature"]


def test_nested_enum_qualified():
    content = "class E { public: enum class Mode { A, B }; };\n"
    syms = _extract_cpp_symbols_ts(content)
    if not syms:
        return
    e = _by_name(syms)["E::Mode"]
    assert e["kind"] == "enum"
    assert "A" in e["signature"] and "B" in e["signature"]


def test_empty_and_garbage_return_empty():
    assert _extract_cpp_symbols_ts("") == []
    assert _extract_cpp_symbols_ts("      \n\n   ") == []
    assert _extract_cpp_symbols_ts("@#$%^&*(") == []
