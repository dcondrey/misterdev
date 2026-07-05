"""Tree-sitter based public-symbol extraction for Rust source."""

from typing import Dict, List, Optional

from ._log import logger


def _extract_rust_symbols_ts(content: str) -> List[Dict[str, str]]:
    """Extract public Rust API using tree-sitter, reusing topography's parser.

    Returns [] when tree-sitter or the Rust grammar is unavailable so the
    caller can fall back to the line-based parser.
    """
    try:
        from misterdev.core.context.topography import _get_ts_parsers

        parser = _get_ts_parsers().get("rust")
    except Exception as e:
        logger.debug(f"tree-sitter rust parser unavailable: {e}")
        return []
    if parser is None:
        return []
    try:
        tree = parser.parse(bytes(content, "utf8"))
    except Exception as e:
        logger.debug(f"tree-sitter parse failed; using line parser: {e}")
        return []
    symbols: List[Dict[str, str]] = []
    _walk_rust_ts(tree.root_node, content, symbols, parent=None)
    return symbols


def _ts_is_pub(node) -> bool:
    return any(c.type == "visibility_modifier" for c in node.children)


def _ts_field_text(node, field: str, content: str) -> str:
    n = node.child_by_field_name(field)
    return content[n.start_byte : n.end_byte] if n else ""


def _ts_decl(node, content: str) -> str:
    """Declaration text up to the body (captures generics/where, drops body)."""
    body = node.child_by_field_name("body")
    end = body.start_byte if body else node.end_byte
    return " ".join(content[node.start_byte : end].split()).strip()


def _walk_rust_ts(
    node, content: str, symbols: List[Dict[str, str]], parent: Optional[str]
):
    t = node.type
    if t == "function_item":
        if _ts_is_pub(node):
            name = _ts_field_text(node, "name", content)
            full = f"{parent}::{name}" if parent else name
            symbols.append(
                {"kind": "pub fn", "name": full, "signature": _ts_decl(node, content)}
            )
        return
    if t == "struct_item" and _ts_is_pub(node):
        name = _ts_field_text(node, "name", content)
        sig = _ts_decl(node, content)
        fields = _ts_pub_members(node, "field_declaration", content)
        if fields:
            sig += " { " + ", ".join(fields) + " }"
        symbols.append({"kind": "pub struct", "name": name, "signature": sig})
        return
    if t == "enum_item" and _ts_is_pub(node):
        name = _ts_field_text(node, "name", content)
        variants = _ts_variant_names(node, content)
        sig = _ts_decl(node, content)
        if variants:
            sig += " { " + ", ".join(variants) + " }"
        symbols.append({"kind": "pub enum", "name": name, "signature": sig})
        return
    if t == "trait_item" and _ts_is_pub(node):
        name = _ts_field_text(node, "name", content)
        methods = _ts_trait_methods(node, content)
        sig = _ts_decl(node, content)
        if methods:
            sig += " { " + "; ".join(methods) + "; }"
        symbols.append({"kind": "pub trait", "name": name, "signature": sig})
        return
    if t == "type_item" and _ts_is_pub(node):
        name = _ts_field_text(node, "name", content)
        text = " ".join(content[node.start_byte : node.end_byte].split()).rstrip(";")
        symbols.append({"kind": "pub type", "name": name, "signature": text})
        return
    if t == "impl_item":
        type_node = node.child_by_field_name("type")
        body = node.child_by_field_name("body")
        if type_node is not None and body is not None:
            impl_name = (
                content[type_node.start_byte : type_node.end_byte].split("<")[0].strip()
            )
            for c in body.children:
                _walk_rust_ts(c, content, symbols, parent=impl_name)
        return
    for c in node.children:
        _walk_rust_ts(c, content, symbols, parent)


def _ts_pub_members(node, child_type: str, content: str) -> List[str]:
    body = node.child_by_field_name("body")
    members = []
    if body is not None:
        for c in body.children:
            if c.type == child_type and any(
                g.type == "visibility_modifier" for g in c.children
            ):
                members.append(
                    " ".join(content[c.start_byte : c.end_byte].split()).rstrip(",")
                )
    return members[:10]


def _ts_variant_names(node, content: str) -> List[str]:
    body = node.child_by_field_name("body")
    names = []
    if body is not None:
        for c in body.children:
            if c.type == "enum_variant":
                n = c.child_by_field_name("name")
                if n is not None:
                    names.append(content[n.start_byte : n.end_byte])
    return names[:15]


def _ts_trait_methods(node, content: str) -> List[str]:
    body = node.child_by_field_name("body")
    methods = []
    if body is not None:
        for c in body.children:
            if c.type in ("function_signature_item", "function_item"):
                b = c.child_by_field_name("body")
                end = b.start_byte if b is not None else c.end_byte
                methods.append(
                    " ".join(content[c.start_byte : end].split()).rstrip(";")
                )
    return methods[:10]
