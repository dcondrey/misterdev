"""Tree-sitter based public-symbol extraction for C source.

C has no visibility keywords; the public surface is the set of non-``static``
externally-visible declarations: function definitions and prototypes,
struct/union/enum definitions, and typedefs. Mirrors ``rust_tree_sitter`` in
shape so the dispatcher can prefer it and fall back to the line heuristic when
the grammar is unavailable.
"""

from typing import Dict, List, Optional

from ._log import logger


def _extract_c_symbols_ts(content: str) -> List[Dict[str, str]]:
    """Extract the public C API using tree-sitter, reusing topography's parser.

    Returns [] when tree-sitter or the C grammar is unavailable so the caller
    can fall back to the generic line-based parser.
    """
    try:
        from misterdev.core.context.topography import _get_ts_parsers

        parser = _get_ts_parsers().get("c")
    except Exception as e:
        logger.debug(f"tree-sitter c parser unavailable: {e}")
        return []
    if parser is None:
        return []
    try:
        tree = parser.parse(bytes(content, "utf8"))
    except Exception as e:
        logger.debug(f"tree-sitter parse failed; using line parser: {e}")
        return []
    symbols: List[Dict[str, str]] = []
    _walk_c_ts(tree.root_node, content, symbols)
    return symbols


def _c_is_static(node) -> bool:
    """True when a declaration carries the ``static`` storage class."""
    for c in node.children:
        if c.type == "storage_class_specifier" and any(
            g.type == "static" for g in c.children
        ):
            return True
    return False


def _c_collapse(content: str, start: int, end: int) -> str:
    return " ".join(content[start:end].split()).strip()


def _c_header(node, content: str) -> str:
    """Declaration text up to the body/terminator (drops ``{ ... }`` and ``;``)."""
    body = node.child_by_field_name("body")
    end = body.start_byte if body is not None else node.end_byte
    return _c_collapse(content, node.start_byte, end).rstrip(";").strip()


def _c_find_function_declarator(node):
    """Descend the ``declarator`` chain to the ``function_declarator``, if any."""
    if node is None:
        return None
    if node.type == "function_declarator":
        return node
    return _c_find_function_declarator(node.child_by_field_name("declarator"))


def _c_declarator_name(node, content: str) -> str:
    """Follow the ``declarator`` chain to the innermost identifier name."""
    while node is not None:
        if node.type in ("identifier", "field_identifier", "type_identifier"):
            return content[node.start_byte : node.end_byte]
        node = node.child_by_field_name("declarator")
    return ""


def _c_function_symbol(node, content: str) -> Optional[Dict[str, str]]:
    if _c_is_static(node):
        return None
    fd = _c_find_function_declarator(node.child_by_field_name("declarator"))
    if fd is None:
        return None
    name = _c_declarator_name(fd.child_by_field_name("declarator"), content)
    if not name:
        return None
    return {"kind": "fn", "name": name, "signature": _c_header(node, content)}


def _c_struct_symbol(node, content: str) -> Optional[Dict[str, str]]:
    body = node.child_by_field_name("body")
    if body is None:  # forward declaration / bare reference, not a definition
        return None
    name_node = node.child_by_field_name("name")
    name = (
        content[name_node.start_byte : name_node.end_byte]
        if name_node is not None
        else ""
    )
    kind = "union" if node.type == "union_specifier" else "struct"
    sig = _c_header(node, content)
    fields = _c_field_names(body, content)
    if fields:
        sig += " { " + ", ".join(fields) + " }"
    return {"kind": kind, "name": name, "signature": sig}


def _c_enum_symbol(node, content: str) -> Optional[Dict[str, str]]:
    body = node.child_by_field_name("body")
    if body is None:
        return None
    name_node = node.child_by_field_name("name")
    name = (
        content[name_node.start_byte : name_node.end_byte]
        if name_node is not None
        else ""
    )
    sig = _c_header(node, content)
    enumerators = _c_enumerator_names(body, content)
    if enumerators:
        sig += " { " + ", ".join(enumerators) + " }"
    return {"kind": "enum", "name": name, "signature": sig}


def _c_typedef_symbol(node, content: str) -> Optional[Dict[str, str]]:
    name = _c_declarator_name(node.child_by_field_name("declarator"), content)
    sig = _c_collapse(content, node.start_byte, node.end_byte).rstrip(";").strip()
    return {"kind": "typedef", "name": name, "signature": sig}


def _walk_c_ts(node, content: str, symbols: List[Dict[str, str]]):
    t = node.type
    if t == "function_definition":
        sym = _c_function_symbol(node, content)
        if sym is not None:
            symbols.append(sym)
        return
    if t == "declaration":
        # A declaration is only part of the API surface when it declares a
        # function (a prototype). Skip static ones and plain variables.
        if _c_find_function_declarator(node.child_by_field_name("declarator")):
            sym = _c_function_symbol(node, content)
            if sym is not None:
                symbols.append(sym)
        return
    if t in ("struct_specifier", "union_specifier"):
        sym = _c_struct_symbol(node, content)
        if sym is not None:
            symbols.append(sym)
        return
    if t == "enum_specifier":
        sym = _c_enum_symbol(node, content)
        if sym is not None:
            symbols.append(sym)
        return
    if t == "type_definition":
        sym = _c_typedef_symbol(node, content)
        if sym is not None:
            symbols.append(sym)
        return
    for c in node.children:
        _walk_c_ts(c, content, symbols)


def _c_field_names(body, content: str) -> List[str]:
    fields = []
    for c in body.children:
        if c.type == "field_declaration":
            fields.append(_c_collapse(content, c.start_byte, c.end_byte).rstrip(";"))
    return fields[:10]


def _c_enumerator_names(body, content: str) -> List[str]:
    names = []
    for c in body.children:
        if c.type == "enumerator":
            n = c.child_by_field_name("name")
            if n is None:
                n = next((g for g in c.children if g.type == "identifier"), None)
            if n is not None:
                names.append(content[n.start_byte : n.end_byte])
    return names[:15]
