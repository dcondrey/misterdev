"""Tree-sitter based public-symbol extraction for Kotlin source.

Kotlin's default visibility is ``public``; this extractor captures the public
API surface (top-level functions, classes, interfaces, objects, enum classes
and their entries, data classes and their constructor params, plus public
member functions/properties) and skips anything marked ``private`` or
``internal``.

The Kotlin tree-sitter grammar emits ``ERROR`` nodes on some perfectly valid
code, so the walker never bails on an ``ERROR`` node -- it descends through it
and extracts whatever valid declaration nodes it can find. Only a hard
``.parse()`` failure (or a missing grammar) yields ``[]`` so the caller can
fall back to the generic line parser.
"""

from typing import Dict, List, Optional

from ._log import logger

# Caps mirror the Rust helpers: keep contracts compact.
_MAX_ENUM_ENTRIES = 20
_MAX_CTOR_PARAMS = 15

_HIDDEN_VISIBILITY = ("private", "internal")
_BODY_TYPES = ("function_body", "class_body", "enum_class_body")


def _extract_kotlin_symbols_ts(content: str) -> List[Dict[str, str]]:
    """Extract public Kotlin API using tree-sitter, reusing topography's parser.

    Returns ``[]`` when tree-sitter or the Kotlin grammar is unavailable, or
    when parsing throws, so the caller can fall back to the line-based parser.
    """
    try:
        from misterdev.core.context.topography import _get_ts_parsers

        parser = _get_ts_parsers().get("kotlin")
    except Exception as e:
        logger.debug(f"tree-sitter kotlin parser unavailable: {e}")
        return []
    if parser is None:
        return []
    try:
        tree = parser.parse(bytes(content, "utf8"))
    except Exception as e:
        logger.debug(f"tree-sitter parse failed; using line parser: {e}")
        return []
    symbols: List[Dict[str, str]] = []
    _walk_kotlin_ts(tree.root_node, content, symbols, parent=None)
    return symbols


def _ts_field_text(node, field: str, content: str) -> str:
    n = node.child_by_field_name(field)
    return content[n.start_byte : n.end_byte] if n else ""


def _kt_name(node, content: str) -> str:
    """Declared name via the ``name`` field, falling back to first identifier."""
    text = _ts_field_text(node, "name", content)
    if text:
        return text
    for c in node.children:
        if c.type in ("identifier", "type_identifier"):
            return content[c.start_byte : c.end_byte]
    return ""


def _kt_ext_name(node, content: str) -> str:
    """Receiver-qualified name for an extension fun ``fun Foo.bar()`` -> ``Foo.bar``."""
    kids = node.children
    for i, c in enumerate(kids):
        if c.type == "." and i > 0:
            recv = content[kids[i - 1].start_byte : kids[i - 1].end_byte].strip()
            for nxt in kids[i + 1 :]:
                if nxt.type == "identifier":
                    method = content[nxt.start_byte : nxt.end_byte]
                    return f"{recv}.{method}" if recv else method
            return ""
    return ""


def _kt_hidden(node) -> bool:
    """True when a declaration is explicitly ``private`` or ``internal``."""
    for c in node.children:
        if c.type != "modifiers":
            continue
        for g in c.children:
            if g.type == "visibility_modifier":
                for tok in g.children:
                    if tok.type in _HIDDEN_VISIBILITY:
                        return True
    return False


def _kt_class_modifiers(node) -> List[str]:
    """Class-level keyword modifiers, e.g. ``data``/``enum``/``sealed``."""
    mods: List[str] = []
    for c in node.children:
        if c.type != "modifiers":
            continue
        for g in c.children:
            if g.type == "class_modifier":
                for tok in g.children:
                    mods.append(tok.type)
    return mods


def _kt_decl(node, content: str) -> str:
    """Declaration text up to the body (keeps params/return, drops body)."""
    cut = node.end_byte
    for c in node.children:
        if c.type in _BODY_TYPES:
            cut = c.start_byte
            break
    return " ".join(content[node.start_byte : cut].split()).strip()


def _kt_property_decl(node, content: str) -> str:
    """Property text up to any initializer (``=`` / delegate ``by``)."""
    cut = node.end_byte
    for c in node.children:
        if c.type in ("=", "by"):
            cut = c.start_byte
            break
    return " ".join(content[node.start_byte : cut].split()).strip()


def _kt_body(node):
    for c in node.children:
        if c.type in ("class_body", "enum_class_body"):
            return c
    return None


def _kt_enum_entries(body, content: str) -> List[str]:
    names: List[str] = []
    for c in body.children:
        if c.type == "enum_entry":
            for g in c.children:
                if g.type == "identifier":
                    names.append(content[g.start_byte : g.end_byte])
                    break
    return names[:_MAX_ENUM_ENTRIES]


def _kt_class_kind(node) -> str:
    """Human-readable kind for a ``class_declaration`` node."""
    mods = _kt_class_modifiers(node)
    if any(c.type == "interface" for c in node.children):
        return "interface"
    if "enum" in mods:
        return "enum class"
    if "data" in mods:
        return "data class"
    if "annotation" in mods:
        return "annotation class"
    if "sealed" in mods:
        return "sealed class"
    return "class"


def _walk_kotlin_ts(
    node, content: str, symbols: List[Dict[str, str]], parent: Optional[str]
):
    t = node.type

    if t == "function_declaration":
        if not _kt_hidden(node):
            name = _kt_ext_name(node, content) or _kt_name(node, content)
            full = f"{parent}.{name}" if parent else name
            symbols.append(
                {"kind": "fun", "name": full, "signature": _kt_decl(node, content)}
            )
        return

    if t == "property_declaration":
        if not _kt_hidden(node):
            name = _kt_property_name(node, content)
            if name:
                full = f"{parent}.{name}" if parent else name
                kind = "var" if _kt_is_var(node) else "val"
                symbols.append(
                    {
                        "kind": kind,
                        "name": full,
                        "signature": _kt_property_decl(node, content),
                    }
                )
        return

    if t == "class_declaration":
        if _kt_hidden(node):
            return
        name = _kt_name(node, content)
        kind = _kt_class_kind(node)
        sig = _kt_decl(node, content)
        body = _kt_body(node)
        if body is not None and body.type == "enum_class_body":
            entries = _kt_enum_entries(body, content)
            if entries:
                sig += " { " + ", ".join(entries) + " }"
        symbols.append({"kind": kind, "name": name, "signature": sig})
        if body is not None:
            for c in body.children:
                _walk_kotlin_ts(c, content, symbols, parent=name)
        return

    if t == "companion_object":
        if _kt_hidden(node):
            return
        name = _kt_name(node, content) or "Companion"
        full = f"{parent}.{name}" if parent else name
        symbols.append(
            {
                "kind": "companion object",
                "name": full,
                "signature": _kt_decl(node, content),
            }
        )
        body = _kt_body(node)
        if body is not None:
            for c in body.children:
                _walk_kotlin_ts(c, content, symbols, parent=parent)
        return

    if t == "type_alias":
        if not _kt_hidden(node):
            name = _kt_name(node, content)
            if name:
                full = f"{parent}.{name}" if parent else name
                symbols.append(
                    {
                        "kind": "typealias",
                        "name": full,
                        "signature": _kt_decl(node, content),
                    }
                )
        return

    if t == "object_declaration":
        if _kt_hidden(node):
            return
        name = _kt_name(node, content)
        symbols.append(
            {"kind": "object", "name": name, "signature": _kt_decl(node, content)}
        )
        body = _kt_body(node)
        if body is not None:
            for c in body.children:
                _walk_kotlin_ts(c, content, symbols, parent=name)
        return

    # Default (incl. ERROR / source_file / package_header): descend.
    for c in node.children:
        _walk_kotlin_ts(c, content, symbols, parent)


def _kt_property_name(node, content: str) -> str:
    for c in node.children:
        if c.type == "variable_declaration":
            for g in c.children:
                if g.type == "identifier":
                    return content[g.start_byte : g.end_byte]
    for c in node.children:
        if c.type == "identifier":
            return content[c.start_byte : c.end_byte]
    return ""


def _kt_is_var(node) -> bool:
    return any(c.type == "var" for c in node.children)
