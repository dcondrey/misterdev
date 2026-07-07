"""Tree-sitter based public-symbol extraction for Swift source."""

from typing import Dict, List, Optional

from ._log import logger

# Swift's declared visibilities, ordered least-to-most restrictive. Anything in
# this set is treated as "hidden" for contract purposes.
_HIDDEN_VIS = ("private", "fileprivate")
# Node children that mark the end of a property's declaration text (the value,
# computed body, or accessor requirements all belong to the "body", not the
# public signature).
_PROP_BODY = (
    "=",
    "computed_property",
    "protocol_property_requirements",
    "willset_didset_block",
)


def _extract_swift_symbols_ts(content: str) -> List[Dict[str, str]]:
    """Extract public Swift API using tree-sitter, reusing topography's parser.

    Returns [] when tree-sitter or the Swift grammar is unavailable so the
    caller can fall back to the line-based parser.
    """
    try:
        from misterdev.core.context.topography import _get_ts_parsers

        parser = _get_ts_parsers().get("swift")
    except Exception as e:
        logger.debug(f"tree-sitter swift parser unavailable: {e}")
        return []
    if parser is None:
        return []
    try:
        tree = parser.parse(bytes(content, "utf8"))
    except Exception as e:
        logger.debug(f"tree-sitter parse failed; using line parser: {e}")
        return []
    symbols: List[Dict[str, str]] = []
    _walk_swift_ts(tree.root_node, content, symbols, parent=None)
    return symbols


def _ts_swift_visibility(node, content: str) -> str:
    """Return the declared visibility keyword, or "" when unmarked (internal)."""
    for c in node.children:
        if c.type == "modifiers":
            for g in c.children:
                if g.type == "visibility_modifier":
                    return content[g.start_byte : g.end_byte].strip()
    return ""


def _ts_swift_field_text(node, field: str, content: str) -> str:
    n = node.child_by_field_name(field)
    return content[n.start_byte : n.end_byte] if n else ""


def _ts_swift_decl(node, content: str) -> str:
    """Declaration text up to the body (drops the `{ ... }` body)."""
    body = node.child_by_field_name("body")
    end = body.start_byte if body else node.end_byte
    return " ".join(content[node.start_byte : end].split()).strip()


def _ts_swift_prop_decl(node, content: str) -> str:
    """Property text up to its value / computed body / accessor requirements."""
    end = node.end_byte
    for c in node.children:
        if c.type in _PROP_BODY:
            end = c.start_byte
            break
    return " ".join(content[node.start_byte : end].split()).strip()


def _is_visible_type(vis: str, parent: Optional[str]) -> bool:
    """Types are kept when public/open, or top-level and not explicitly hidden.

    Exercise code often omits modifiers, so top-level type declarations are
    included even when unmarked (internal), but never when private/fileprivate.
    """
    if vis in ("public", "open"):
        return True
    return parent is None and vis not in _HIDDEN_VIS


def _walk_swift_ts(
    node, content: str, symbols: List[Dict[str, str]], parent: Optional[str]
):
    t = node.type

    if t == "function_declaration":
        if _ts_swift_visibility(node, content) in ("public", "open"):
            name = _ts_swift_field_text(node, "name", content)
            full = f"{parent}.{name}" if parent else name
            symbols.append(
                {
                    "kind": "func",
                    "name": full,
                    "signature": _ts_swift_decl(node, content),
                }
            )
        return

    if t == "property_declaration":
        # Nested public properties are embedded in their type's signature; only
        # emit stand-alone (top-level global) public properties here.
        if parent is None and _ts_swift_visibility(node, content) in (
            "public",
            "open",
        ):
            name = _ts_swift_field_text(node, "name", content)
            mut = _ts_swift_mutability(node, content)
            symbols.append(
                {
                    "kind": mut,
                    "name": name,
                    "signature": _ts_swift_prop_decl(node, content),
                }
            )
        return

    if t == "class_declaration":
        vis = _ts_swift_visibility(node, content)
        if not _is_visible_type(vis, parent):
            return
        name = _ts_swift_field_text(node, "name", content)
        dk = node.child_by_field_name("declaration_kind")
        kind = content[dk.start_byte : dk.end_byte] if dk is not None else "type"
        sig = _ts_swift_decl(node, content)
        if kind == "enum":
            cases = _ts_swift_enum_cases(node, content)
            if cases:
                sig += " { " + ", ".join(cases) + " }"
        else:
            props = _ts_swift_pub_properties(node, content)
            if props:
                sig += " { " + "; ".join(props) + " }"
        symbols.append({"kind": kind, "name": name, "signature": sig})
        # Walk the body so public methods are emitted as `Type.method`.
        body = node.child_by_field_name("body")
        if body is not None:
            for c in body.children:
                _walk_swift_ts(c, content, symbols, parent=name)
        return

    if t == "protocol_declaration":
        vis = _ts_swift_visibility(node, content)
        if not _is_visible_type(vis, parent):
            return
        name = _ts_swift_field_text(node, "name", content)
        sig = _ts_swift_decl(node, content)
        reqs = _ts_swift_protocol_requirements(node, content)
        if reqs:
            sig += " { " + "; ".join(reqs) + "; }"
        symbols.append({"kind": "protocol", "name": name, "signature": sig})
        return

    if t == "init_declaration":
        # Initializers only appear inside a type/extension; emit only public
        # ones as `Type.init`, dropping the body like any other method.
        if _ts_swift_visibility(node, content) in ("public", "open"):
            full = f"{parent}.init" if parent else "init"
            symbols.append(
                {
                    "kind": "init",
                    "name": full,
                    "signature": _ts_swift_decl(node, content),
                }
            )
        return

    if t == "subscript_declaration":
        # subscripts have no name field; nest as `Type.subscript`. The computed
        # body / accessor block is dropped via the property-body markers.
        if _ts_swift_visibility(node, content) in ("public", "open"):
            full = f"{parent}.subscript" if parent else "subscript"
            symbols.append(
                {
                    "kind": "subscript",
                    "name": full,
                    "signature": _ts_swift_prop_decl(node, content),
                }
            )
        return

    if t == "typealias_declaration":
        # Type aliases follow the type-visibility rule: public/open always, or
        # top-level even when unmarked. Nested aliases nest as `Type.Alias`.
        vis = _ts_swift_visibility(node, content)
        if _is_visible_type(vis, parent):
            name = _ts_swift_field_text(node, "name", content)
            full = f"{parent}.{name}" if parent else name
            symbols.append(
                {
                    "kind": "typealias",
                    "name": full,
                    "signature": _ts_swift_decl(node, content),
                }
            )
        return

    for c in node.children:
        _walk_swift_ts(c, content, symbols, parent)


def _ts_swift_mutability(node, content: str) -> str:
    """ "var" or "let" for a property_declaration; defaults to "var"."""
    for c in node.children:
        if c.type == "value_binding_pattern":
            for g in c.children:
                if g.type in ("var", "let"):
                    return g.type
    return "var"


def _ts_swift_enum_cases(node, content: str) -> List[str]:
    body = node.child_by_field_name("body")
    names: List[str] = []
    if body is not None:
        for c in body.children:
            if c.type == "enum_entry":
                for g in c.children:
                    if g.type == "simple_identifier":
                        names.append(content[g.start_byte : g.end_byte])
    return names[:15]


def _ts_swift_pub_properties(node, content: str) -> List[str]:
    body = node.child_by_field_name("body")
    props: List[str] = []
    if body is not None:
        for c in body.children:
            if c.type == "property_declaration" and _ts_swift_visibility(
                c, content
            ) in ("public", "open"):
                props.append(_ts_swift_prop_decl(c, content))
    return props[:10]


def _ts_swift_protocol_requirements(node, content: str) -> List[str]:
    body = node.child_by_field_name("body")
    reqs: List[str] = []
    if body is not None:
        for c in body.children:
            if c.type == "protocol_function_declaration":
                b = c.child_by_field_name("body")
                end = b.start_byte if b is not None else c.end_byte
                reqs.append(" ".join(content[c.start_byte : end].split()))
            elif c.type in ("protocol_property_declaration", "property_declaration"):
                reqs.append(_ts_swift_prop_decl(c, content))
            elif c.type == "associatedtype_declaration":
                reqs.append(" ".join(content[c.start_byte : c.end_byte].split()))
    return reqs[:10]
