"""Tree-sitter based public-API extraction for C++ source.

Mirrors ``rust_tree_sitter`` in shape: a single public entry point
``_extract_cpp_symbols_ts`` reuses topography's cached parser, and returns
``[]`` on any failure so the caller can fall back to the line heuristic.
"""

from typing import Dict, List, Optional

from ._log import logger

# Caps mirror the Rust helpers so a pathological header cannot blow up a prompt.
_MAX_FIELDS = 10
_MAX_METHODS = 15
_MAX_ENUMERATORS = 20
_MAX_NESTED = 8

# Record/enum specifiers that, when they appear inside a class/struct body,
# are nested type definitions rather than data members.
_NESTED_TYPES = frozenset({"class_specifier", "struct_specifier", "enum_specifier"})

# Substantive top-level declaration node types we turn into symbols.
_DECL_TYPES = frozenset(
    {
        "function_definition",
        "declaration",
        "class_specifier",
        "struct_specifier",
        "enum_specifier",
        "type_definition",
        "alias_declaration",
    }
)
# Container nodes we descend into without emitting a symbol of their own.
_CONTAINER_TYPES = frozenset(
    {"translation_unit", "declaration_list", "linkage_specification"}
)
# Declarator wrappers around a name (pointers/references/parens).
_WRAP_DECLARATORS = frozenset(
    {"pointer_declarator", "reference_declarator", "parenthesized_declarator"}
)
_NAME_NODES = frozenset(
    {
        "identifier",
        "field_identifier",
        "type_identifier",
        "qualified_identifier",
        "destructor_name",
        "operator_name",
        "template_function",
    }
)


def _extract_cpp_symbols_ts(content: str) -> List[Dict[str, str]]:
    """Extract the public C++ API surface using tree-sitter.

    Returns ``[]`` when tree-sitter or the C++ grammar is unavailable, or when
    parsing fails, so the caller can fall back to the line-based parser.
    """
    try:
        from misterdev.core.context.topography import _get_ts_parsers

        parser = _get_ts_parsers().get("cpp")
    except Exception as e:
        logger.debug(f"tree-sitter cpp parser unavailable: {e}")
        return []
    if parser is None:
        return []
    try:
        tree = parser.parse(bytes(content, "utf8"))
    except Exception as e:
        logger.debug(f"tree-sitter cpp parse failed; using line parser: {e}")
        return []
    symbols: List[Dict[str, str]] = []
    try:
        _walk_cpp_ts(tree.root_node, content, symbols)
    except Exception as e:  # tolerate any unexpected grammar shape
        logger.debug(f"tree-sitter cpp walk failed: {e}")
        return symbols
    return symbols


def _text(node, content: str) -> str:
    return content[node.start_byte : node.end_byte] if node is not None else ""


def _collapse(text: str) -> str:
    """Whitespace-collapse and drop a trailing declaration terminator."""
    return " ".join(text.split()).strip().rstrip(";").strip()


def _sig(node, content: str, sig_start: Optional[int] = None) -> str:
    """Signature text: from ``sig_start`` (or node start) up to the body/``;``.

    ``sig_start`` lets a templated declaration prepend its ``template<...>``.
    """
    start = node.start_byte if sig_start is None else sig_start
    body = node.child_by_field_name("body")
    end = body.start_byte if body is not None else node.end_byte
    return _collapse(content[start:end])


def _child_declarator(node):
    """Inner declarator of a wrapper node.

    ``reference_declarator`` (unlike ``pointer_declarator``) does not tag its
    inner declarator with the ``declarator`` field in tree-sitter-cpp, so fall
    back to the first declarator-shaped child when the field is absent.
    """
    d = node.child_by_field_name("declarator")
    if d is not None:
        return d
    for c in node.children:
        if (
            c.type == "function_declarator"
            or c.type in _WRAP_DECLARATORS
            or c.type in _NAME_NODES
        ):
            return c
    return None


def _find_function_declarator(node):
    """Descend a declarator chain to the ``function_declarator``, if any."""
    if node is None:
        return None
    if node.type == "function_declarator":
        return node
    if node.type in _WRAP_DECLARATORS:
        return _find_function_declarator(_child_declarator(node))
    return None


def _declarator_name(node, content: str) -> str:
    """Resolve the identifier a declarator ultimately names."""
    if node is None:
        return ""
    t = node.type
    if t in _NAME_NODES:
        return _text(node, content).strip()
    if t == "function_declarator" or t in _WRAP_DECLARATORS:
        return _declarator_name(_child_declarator(node), content)
    inner = _child_declarator(node)
    return _declarator_name(inner, content) if inner is not None else ""


def _function_name(node, content: str) -> str:
    """Name of a function_definition / prototype declaration."""
    fdecl = _find_function_declarator(node.child_by_field_name("declarator"))
    if fdecl is None:
        return ""
    return _declarator_name(fdecl.child_by_field_name("declarator"), content)


def _walk_cpp_ts(node, content: str, symbols: List[Dict[str, str]]):
    for child in node.children:
        t = child.type
        if t == "template_declaration":
            inner = _template_inner(child)
            if inner is not None:
                _emit(inner, content, symbols, sig_start=child.start_byte)
        elif t == "namespace_definition":
            body = child.child_by_field_name("body")
            if body is not None:
                _walk_cpp_ts(body, content, symbols)
        elif t in _DECL_TYPES:
            _emit(child, content, symbols, sig_start=None)
        elif t in _CONTAINER_TYPES:
            _walk_cpp_ts(child, content, symbols)


def _template_inner(node):
    """The substantive declaration a template_declaration wraps."""
    for c in node.children:
        if c.type in _DECL_TYPES:
            return c
    return None


def _emit(node, content: str, symbols: List[Dict[str, str]], sig_start: Optional[int]):
    t = node.type
    if t == "function_definition":
        _emit_function(node, content, symbols, sig_start)
    elif t == "declaration":
        # Only function prototypes are part of the API surface here.
        if _find_function_declarator(node.child_by_field_name("declarator")):
            _emit_function(node, content, symbols, sig_start)
    elif t in ("class_specifier", "struct_specifier"):
        _emit_record(node, content, symbols, sig_start)
    elif t == "enum_specifier":
        _emit_enum(node, content, symbols, sig_start)
    elif t == "type_definition":
        _emit_typedef(node, content, symbols, sig_start)
    elif t == "alias_declaration":
        _emit_using(node, content, symbols, sig_start)


def _emit_function(
    node, content: str, symbols: List[Dict[str, str]], sig_start: Optional[int]
):
    name = _function_name(node, content)
    if not name:
        return
    symbols.append(
        {"kind": "function", "name": name, "signature": _sig(node, content, sig_start)}
    )


def _emit_record(
    node,
    content: str,
    symbols: List[Dict[str, str]],
    sig_start: Optional[int],
    qual: str = "",
):
    name = _text(node.child_by_field_name("name"), content).strip()
    if not name:
        return
    full = qual + name
    kind = "struct" if node.type == "struct_specifier" else "class"
    body = node.child_by_field_name("body")
    fields, methods, nested = _record_members(body, content, is_struct=kind == "struct")
    sig = _sig(node, content, sig_start)
    if fields:
        sig += " { " + ", ".join(fields) + " }"
    symbols.append({"kind": kind, "name": full, "signature": sig})
    for method_node, method_start in methods:
        mname = _function_name(method_node, content)
        if not mname:
            continue
        symbols.append(
            {
                "kind": "method",
                "name": f"{full}::{mname}",
                "signature": _sig(method_node, content, method_start),
            }
        )
    for rec in nested:
        if rec.type == "enum_specifier":
            _emit_enum(rec, content, symbols, None, qual=f"{full}::")
        else:
            _emit_record(rec, content, symbols, None, qual=f"{full}::")


def _record_members(body, content: str, is_struct: bool):
    """Public fields (as text), method nodes, and nested type nodes.

    struct members default public; class members default private.
    """
    fields: List[str] = []
    methods = []
    nested = []
    if body is None:
        return fields, methods, nested
    access = "public" if is_struct else "private"
    for c in body.children:
        if c.type == "access_specifier":
            access = _text(c, content).strip() or access
            continue
        if access != "public":
            continue
        if c.type in _NESTED_TYPES and c.child_by_field_name("body") is not None:
            if len(nested) < _MAX_NESTED:
                nested.append(c)
        elif c.type == "function_definition":
            if len(methods) < _MAX_METHODS:
                methods.append((c, None))
        elif c.type == "field_declaration":
            inner = _nested_type_child(c)
            if inner is not None:
                if len(nested) < _MAX_NESTED:
                    nested.append(inner)
            elif _find_function_declarator(c.child_by_field_name("declarator")):
                if len(methods) < _MAX_METHODS:
                    methods.append((c, None))
            elif len(fields) < _MAX_FIELDS:
                fields.append(_collapse(_text(c, content)).rstrip(",").strip())
    return fields, methods, nested


def _nested_type_child(field_decl):
    """A nested type definition wrapped in a ``field_declaration``, if any.

    ``class Inner { ... };`` inside a body parses as a field_declaration whose
    type child is a record/enum specifier carrying its own body.
    """
    for c in field_decl.children:
        if c.type in _NESTED_TYPES and c.child_by_field_name("body") is not None:
            return c
    return None


def _emit_enum(
    node,
    content: str,
    symbols: List[Dict[str, str]],
    sig_start: Optional[int],
    qual: str = "",
):
    name = _text(node.child_by_field_name("name"), content).strip()
    if not name:
        return
    sig = _sig(node, content, sig_start)
    enumerators = _enumerators(node.child_by_field_name("body"), content)
    if enumerators:
        sig += " { " + ", ".join(enumerators) + " }"
    symbols.append({"kind": "enum", "name": qual + name, "signature": sig})


def _enumerators(body, content: str) -> List[str]:
    names: List[str] = []
    if body is None:
        return names
    for c in body.children:
        if c.type == "enumerator":
            n = c.child_by_field_name("name")
            if n is not None:
                names.append(_text(n, content).strip())
                if len(names) >= _MAX_ENUMERATORS:
                    break
    return names


def _emit_typedef(
    node, content: str, symbols: List[Dict[str, str]], sig_start: Optional[int]
):
    name = _declarator_name(node.child_by_field_name("declarator"), content)
    if not name:
        return
    symbols.append(
        {"kind": "typedef", "name": name, "signature": _sig(node, content, sig_start)}
    )


def _emit_using(
    node, content: str, symbols: List[Dict[str, str]], sig_start: Optional[int]
):
    name = _text(node.child_by_field_name("name"), content).strip()
    if not name:
        return
    symbols.append(
        {"kind": "using", "name": name, "signature": _sig(node, content, sig_start)}
    )
