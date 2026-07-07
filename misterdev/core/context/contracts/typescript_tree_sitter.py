"""Tree-sitter based public-symbol extraction for TypeScript source."""

from typing import Dict, List

from ._log import logger

# Inner declaration node types produced under an `export_statement` wrapper.
_DECL_TYPES = {
    "function_declaration",
    "generator_function_declaration",
    "function_signature",  # overload signature: `export function f(x): void;`
    "class_declaration",
    "abstract_class_declaration",
    "interface_declaration",
    "type_alias_declaration",
    "lexical_declaration",
    "variable_declaration",
    "enum_declaration",
    "internal_module",  # `export namespace NS { ... }`
    "module",  # `export module M { ... }`
    "ambient_declaration",  # `export declare function/class ...`
}


def _extract_typescript_symbols_ts(content: str) -> List[Dict[str, str]]:
    """Extract the exported TypeScript API using tree-sitter.

    Reuses topography's parser registry (key ``"typescript"``). Returns [] when
    tree-sitter or the TypeScript grammar is unavailable so the caller can fall
    back to the generic line-based parser.
    """
    try:
        from misterdev.core.context.topography import _get_ts_parsers

        parser = _get_ts_parsers().get("typescript")
    except Exception as e:
        logger.debug(f"tree-sitter typescript parser unavailable: {e}")
        return []
    if parser is None:
        return []
    try:
        tree = parser.parse(bytes(content, "utf8"))
    except Exception as e:
        logger.debug(f"tree-sitter parse failed; using line parser: {e}")
        return []
    symbols: List[Dict[str, str]] = []
    for child in tree.root_node.children:
        if child.type == "export_statement":
            _emit_export(child, content, symbols)
    return symbols


def _ts_field_text(node, field: str, content: str) -> str:
    n = node.child_by_field_name(field)
    return content[n.start_byte : n.end_byte] if n else ""


def _collapse(content: str, start: int, end: int) -> str:
    return " ".join(content[start:end].split()).strip()


def _ts_decl(node, content: str) -> str:
    """Declaration text up to the body (drops the body block)."""
    body = node.child_by_field_name("body")
    end = body.start_byte if body is not None else node.end_byte
    return _collapse(content, node.start_byte, end)


def _is_default(export_node) -> bool:
    return any(c.type == "default" for c in export_node.children)


def _inner_decl(export_node):
    decl = export_node.child_by_field_name("declaration")
    if decl is not None:
        return decl
    for c in export_node.children:
        if c.type in _DECL_TYPES:
            return c
    return None


def _child_of_type(node, type_name: str):
    for c in node.children:
        if c.type == type_name:
            return c
    return None


def _emit_export(export_node, content: str, symbols: List[Dict[str, str]]):
    clause = _child_of_type(export_node, "export_clause")
    if clause is not None:
        _emit_export_clause(export_node, clause, content, symbols)
        return
    ns_export = _child_of_type(export_node, "namespace_export")
    has_star = any(c.type == "*" for c in export_node.children)
    if ns_export is not None or has_star:
        _emit_star_reexport(export_node, ns_export, content, symbols)
        return
    decl = _inner_decl(export_node)
    if decl is None:
        if _is_default(export_node):
            _emit_default_expr(export_node, content, symbols)
        return
    prefix = "export default" if _is_default(export_node) else "export"
    _emit_decl(decl, content, symbols, prefix)


def _reexport_kind(export_node) -> str:
    return "re-export" if _child_of_type(export_node, "from") is not None else "export"


def _emit_export_clause(export_node, clause, content, symbols):
    sig = _collapse(content, export_node.start_byte, export_node.end_byte).rstrip(";")
    kind = _reexport_kind(export_node)
    for spec in clause.children:
        if spec.type != "export_specifier":
            continue
        alias = spec.child_by_field_name("alias")
        name_node = alias if alias is not None else spec.child_by_field_name("name")
        if name_node is None:
            continue
        symbols.append(
            {
                "kind": kind,
                "name": content[name_node.start_byte : name_node.end_byte],
                "signature": sig,
            }
        )


def _emit_star_reexport(export_node, ns_export, content, symbols):
    sig = _collapse(content, export_node.start_byte, export_node.end_byte).rstrip(";")
    if ns_export is not None:
        ident = _child_of_type(ns_export, "identifier")
        name = content[ident.start_byte : ident.end_byte] if ident is not None else "*"
    else:
        src = _child_of_type(export_node, "string")
        name = (
            _collapse(content, src.start_byte, src.end_byte).strip("'\"")
            if src is not None
            else "*"
        )
    symbols.append(
        {"kind": f"{_reexport_kind(export_node)} *", "name": name, "signature": sig}
    )


def _emit_default_expr(export_node, content, symbols):
    sig = _collapse(content, export_node.start_byte, export_node.end_byte).rstrip(";")
    ident = _child_of_type(export_node, "identifier")
    name = (
        content[ident.start_byte : ident.end_byte] if ident is not None else "default"
    )
    symbols.append({"kind": "export default", "name": name, "signature": sig})


def _member_is_public(node, content: str) -> bool:
    for c in node.children:
        if c.type == "accessibility_modifier" and content[
            c.start_byte : c.end_byte
        ] in ("private", "protected"):
            return False
    name = node.child_by_field_name("name")
    if name is not None and name.type == "private_property_identifier":
        return False
    return True


def _decl_name(node, content: str) -> str:
    n = node.child_by_field_name("name")
    if n is not None:
        return content[n.start_byte : n.end_byte]
    for c in node.children:
        if c.type in ("identifier", "type_identifier", "property_identifier"):
            return content[c.start_byte : c.end_byte]
    return ""


def _emit_decl(decl, content: str, symbols: List[Dict[str, str]], prefix: str):
    t = decl.type

    if t == "ambient_declaration":
        inner = None
        for c in decl.children:
            if c.type in _DECL_TYPES:
                inner = c
                break
        if inner is not None:
            _emit_decl(inner, content, symbols, f"{prefix} declare")
        return

    if t in (
        "function_declaration",
        "generator_function_declaration",
        "function_signature",
    ):
        name = _decl_name(decl, content) or "default"
        symbols.append(
            {
                "kind": f"{prefix} function",
                "name": name,
                "signature": _ts_decl(decl, content).rstrip(";"),
            }
        )
        return

    if t in ("internal_module", "module"):
        name = _decl_name(decl, content) or "default"
        body = _child_of_type(decl, "statement_block")
        end = body.start_byte if body is not None else decl.end_byte
        keyword = "namespace" if t == "internal_module" else "module"
        symbols.append(
            {
                "kind": f"{prefix} {keyword}",
                "name": name,
                "signature": _collapse(content, decl.start_byte, end),
            }
        )
        _emit_namespace_members(body, content, symbols, name)
        return

    if t in ("class_declaration", "abstract_class_declaration"):
        name = _ts_field_text(decl, "name", content) or "default"
        symbols.append(
            {
                "kind": f"{prefix} class",
                "name": name,
                "signature": _ts_decl(decl, content),
            }
        )
        _emit_class_members(decl, content, symbols, name)
        return

    if t == "interface_declaration":
        name = _ts_field_text(decl, "name", content) or "default"
        symbols.append(
            {
                "kind": f"{prefix} interface",
                "name": name,
                "signature": _ts_decl(decl, content),
            }
        )
        _emit_interface_members(decl, content, symbols, name)
        return

    if t == "type_alias_declaration":
        name = _ts_field_text(decl, "name", content) or "default"
        symbols.append(
            {
                "kind": f"{prefix} type",
                "name": name,
                "signature": _collapse(content, decl.start_byte, decl.end_byte).rstrip(
                    ";"
                ),
            }
        )
        return

    if t == "enum_declaration":
        name = _ts_field_text(decl, "name", content) or "default"
        sig = _ts_decl(decl, content)
        variants = _enum_members(decl, content)
        if variants:
            sig += " { " + ", ".join(variants) + " }"
        symbols.append({"kind": f"{prefix} enum", "name": name, "signature": sig})
        return

    if t in ("lexical_declaration", "variable_declaration"):
        keyword = "const"
        for c in decl.children:
            if c.type in ("const", "let", "var"):
                keyword = c.type
                break
        for c in decl.children:
            if c.type == "variable_declarator":
                name = _ts_field_text(c, "name", content) or "default"
                symbols.append(
                    {
                        "kind": f"{prefix} {keyword}",
                        "name": name,
                        "signature": _collapse(
                            content, decl.start_byte, decl.end_byte
                        ).rstrip(";"),
                    }
                )
        return


def _emit_class_members(decl, content: str, symbols: List[Dict[str, str]], parent: str):
    body = decl.child_by_field_name("body")
    if body is None:
        return
    count = 0
    for c in body.children:
        if count >= 20:
            break
        if c.type not in (
            "method_definition",
            "method_signature",
            "abstract_method_signature",
        ):
            continue
        if not _member_is_public(c, content):
            continue
        name = _ts_field_text(c, "name", content)
        if not name:
            continue
        sig = (
            _ts_decl(c, content)
            if c.type == "method_definition"
            else _collapse(content, c.start_byte, c.end_byte).rstrip(";")
        )
        symbols.append(
            {
                "kind": "method",
                "name": f"{parent}.{name}",
                "signature": sig,
            }
        )
        count += 1


def _emit_namespace_members(
    body, content: str, symbols: List[Dict[str, str]], parent: str
):
    if body is None:
        return
    for c in body.children:
        if c.type != "export_statement":
            continue
        nested: List[Dict[str, str]] = []
        _emit_export(c, content, nested)
        for s in nested[:20]:
            s["name"] = f"{parent}.{s['name']}"
            symbols.append(s)


def _emit_interface_members(
    decl, content: str, symbols: List[Dict[str, str]], parent: str
):
    body = decl.child_by_field_name("body")
    if body is None:
        return
    count = 0
    for c in body.children:
        if count >= 20:
            break
        if c.type not in ("method_signature", "property_signature"):
            continue
        name = _ts_field_text(c, "name", content)
        if not name:
            continue
        kind = "method" if c.type == "method_signature" else "property"
        symbols.append(
            {
                "kind": kind,
                "name": f"{parent}.{name}",
                "signature": _collapse(content, c.start_byte, c.end_byte).rstrip(";"),
            }
        )
        count += 1


def _enum_members(decl, content: str) -> List[str]:
    body = decl.child_by_field_name("body")
    names: List[str] = []
    if body is None:
        return names
    for c in body.children:
        if c.type == "property_identifier":
            names.append(content[c.start_byte : c.end_byte])
        elif c.type == "enum_assignment":
            n = c.child_by_field_name("name")
            if n is not None:
                names.append(content[n.start_byte : n.end_byte])
    return names[:15]
