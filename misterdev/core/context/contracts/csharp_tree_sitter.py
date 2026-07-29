"""Tree-sitter based public-symbol extraction for C# source."""

from typing import Dict, List, Optional

from ._log import logger

# C# type declarations we surface, mapped to the `kind` we report.
_TYPE_KINDS = {
    "class_declaration": "class",
    "interface_declaration": "interface",
    "struct_declaration": "struct",
    "record_declaration": "record",
    "enum_declaration": "enum",
}
# Modifiers that, when present, mark a declaration as part of the public surface.
_VISIBLE_MODS = ("public", "protected")
# Modifiers that explicitly hide a declaration (used only for interface members,
# whose default is public).
_HIDDEN_MODS = ("private", "internal")


def _extract_csharp_symbols_ts(content: str) -> List[Dict[str, str]]:
    """Extract public C# API using tree-sitter, reusing topography's parser.

    Returns [] when tree-sitter or the C# grammar is unavailable so the caller
    can fall back to the generic line-based parser.
    """
    try:
        from misterdev.core.context.topography import _get_ts_parsers

        parser = _get_ts_parsers().get("csharp")
    except Exception as e:
        logger.debug(f"tree-sitter csharp parser unavailable: {e}")
        return []
    if parser is None:
        return []
    try:
        tree = parser.parse(bytes(content, "utf8"))
    except Exception as e:
        logger.debug(f"tree-sitter parse failed; using line parser: {e}")
        return []
    symbols: List[Dict[str, str]] = []
    _walk_csharp_ts(
        tree.root_node, content, symbols, parent=None, implicit_public=False
    )
    return symbols


def _cs_modifiers(node, content: str) -> List[str]:
    """Collect the modifier keywords declared directly on a node."""
    mods: List[str] = []
    for c in node.children:
        if c.type == "modifier":
            mods.append(content[c.start_byte : c.end_byte].strip())
    return mods


def _cs_is_public(node, content: str, implicit_public: bool = False) -> bool:
    """True when a declaration belongs to the public surface.

    `public`/`protected` (incl. `protected internal`) are always visible. When
    `implicit_public` (interface members), an unmarked member is visible unless
    it carries an explicit `private`/`internal`.
    """
    mods = _cs_modifiers(node, content)
    if any(m in _VISIBLE_MODS for m in mods):
        return True
    if implicit_public and not any(m in _HIDDEN_MODS for m in mods):
        return True
    return False


def _cs_name(node, content: str) -> str:
    """The declaration's name via the `name` field, with an identifier fallback."""
    n = node.child_by_field_name("name")
    if n is None:
        for c in node.children:
            if c.type == "identifier":
                n = c
                break
    return content[n.start_byte : n.end_byte] if n is not None else ""


def _cs_member_sig(node, content: str) -> str:
    """Member text up to its body; drops `{ ... }` / `=> expr` and trailing `;`."""
    body = node.child_by_field_name("body")
    end = body.start_byte if body is not None else node.end_byte
    text = " ".join(content[node.start_byte : end].split()).strip()
    return text[:-1].strip() if text.endswith(";") else text


def _cs_type_sig(node, content: str) -> str:
    """Type header up to its member/enum body; drops the body and trailing `;`."""
    end = node.end_byte
    for c in node.children:
        if c.type in ("declaration_list", "enum_member_declaration_list"):
            end = c.start_byte
            break
    text = " ".join(content[node.start_byte : end].split()).strip()
    return text[:-1].strip() if text.endswith(";") else text


def _cs_enum_members(node, content: str) -> List[str]:
    """Enum member names (always public), capped."""
    names: List[str] = []
    for c in node.children:
        if c.type == "enum_member_declaration_list":
            for m in c.children:
                if m.type == "enum_member_declaration":
                    names.append(_cs_name(m, content))
    return names[:20]


def _cs_pub_member_sigs(node, content: str, implicit_public: bool) -> List[str]:
    """Public method/property/field signatures declared in a type's body."""
    body = None
    for c in node.children:
        if c.type == "declaration_list":
            body = c
            break
    if body is None:
        return []
    sigs: List[str] = []
    for c in body.children:
        if c.type in (
            "method_declaration",
            "property_declaration",
            "field_declaration",
        ) and _cs_is_public(c, content, implicit_public):
            sigs.append(_cs_member_sig(c, content))
    return sigs[:15]


def _walk_csharp_ts(
    node,
    content: str,
    symbols: List[Dict[str, str]],
    parent: Optional[str],
    implicit_public: bool,
):
    t = node.type

    kind = _TYPE_KINDS.get(t)
    if kind is not None:
        if not _cs_is_public(node, content, implicit_public):
            return
        name = _cs_name(node, content)
        full = f"{parent}.{name}" if parent else name
        sig = _cs_type_sig(node, content)
        if kind == "enum":
            members = _cs_enum_members(node, content)
            if members:
                sig += " { " + ", ".join(members) + " }"
        else:
            members = _cs_pub_member_sigs(node, content, kind == "interface")
            if members:
                sig += " { " + "; ".join(members) + " }"
        symbols.append({"kind": kind, "name": full, "signature": sig})
        # Walk the body so public members surface as `Type.Member`.
        body = node.child_by_field_name("body")
        if body is not None:
            child_implicit = kind == "interface"
            for c in body.children:
                _walk_csharp_ts(c, content, symbols, full, child_implicit)
        return

    if t == "method_declaration":
        if parent is not None and _cs_is_public(node, content, implicit_public):
            symbols.append(
                {
                    "kind": "method",
                    "name": f"{parent}.{_cs_name(node, content)}",
                    "signature": _cs_member_sig(node, content),
                }
            )
        return

    if t == "property_declaration":
        if parent is not None and _cs_is_public(node, content, implicit_public):
            symbols.append(
                {
                    "kind": "property",
                    "name": f"{parent}.{_cs_name(node, content)}",
                    "signature": _cs_member_sig(node, content),
                }
            )
        return

    for c in node.children:
        _walk_csharp_ts(c, content, symbols, parent, implicit_public)
