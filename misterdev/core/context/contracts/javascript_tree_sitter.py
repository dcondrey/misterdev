"""Tree-sitter based public-symbol extraction for JavaScript source."""

from typing import Dict, List

from ._log import logger

# Caps mirror the Rust helpers: keep contracts compact for prompt injection.
_MAX_METHODS = 50
_MAX_SIG_LEN = 200


def _extract_javascript_symbols_ts(content: str) -> List[Dict[str, str]]:
    """Extract the exported JavaScript API using tree-sitter.

    Captures ``export function``/``class``/``const``/``let``/``var`` and
    ``export default`` declarations; class methods are nested under their
    class as ``Class.method``. Non-exported top-level declarations are
    skipped.

    Returns ``[]`` when tree-sitter or the JavaScript grammar is unavailable
    so the caller can fall back to the generic line-based parser.
    """
    try:
        from misterdev.core.context.topography import _get_ts_parsers

        parser = _get_ts_parsers().get("javascript")
    except Exception as e:
        logger.debug(f"tree-sitter javascript parser unavailable: {e}")
        return []
    if parser is None:
        return []
    try:
        tree = parser.parse(bytes(content, "utf8"))
    except Exception as e:
        logger.debug(f"tree-sitter parse failed; using line parser: {e}")
        return []
    symbols: List[Dict[str, str]] = []
    _walk_js_ts(tree.root_node, content, symbols)
    return symbols


def _js_field_text(node, field: str, content: str) -> str:
    n = node.child_by_field_name(field)
    return content[n.start_byte : n.end_byte] if n else ""


def _js_collapse(text: str) -> str:
    return " ".join(text.split()).strip()


def _js_decl(node, content: str) -> str:
    """Declaration text up to the body (drops the statement block)."""
    body = node.child_by_field_name("body")
    end = body.start_byte if body else node.end_byte
    return _js_collapse(content[node.start_byte : end])


def _js_truncate(text: str) -> str:
    return text if len(text) <= _MAX_SIG_LEN else text[:_MAX_SIG_LEN].rstrip() + " ..."


def _walk_js_ts(node, content: str, symbols: List[Dict[str, str]]):
    for child in node.children:
        t = child.type
        if t == "export_statement":
            _emit_export(child, content, symbols)
        elif t == "expression_statement":
            _emit_commonjs(child, content, symbols)
        else:
            # Descend into non-export containers (e.g. wrapping blocks) but do
            # not surface their inner top-level declarations: only the exported
            # surface is a contract.
            continue


def _emit_export(node, content: str, symbols: List[Dict[str, str]]):
    is_default = any(c.type == "default" for c in node.children)
    handled = False
    for c in node.children:
        if c.type in ("function_declaration", "generator_function_declaration"):
            _emit_function(c, content, symbols, is_default)
            handled = True
        elif c.type == "class_declaration":
            _emit_class(c, content, symbols, is_default)
            handled = True
        elif c.type in ("lexical_declaration", "variable_declaration"):
            _emit_variables(c, content, symbols)
            handled = True
    if is_default and not handled:
        # `export default <expression>` (identifier, object, arrow fn, ...).
        value = node.child_by_field_name("value")
        if value is None:
            for c in node.children:
                if c.type not in ("export", "default", ";"):
                    value = c
        if value is not None:
            symbols.append(
                {
                    "kind": "export default",
                    "name": "default",
                    "signature": _js_truncate(
                        _js_collapse(content[value.start_byte : value.end_byte])
                    ),
                }
            )


def _emit_function(node, content: str, symbols: List[Dict[str, str]], is_default: bool):
    name = _js_field_text(node, "name", content) or "default"
    kind = "export default function" if is_default else "export function"
    symbols.append({"kind": kind, "name": name, "signature": _js_decl(node, content)})


def _emit_class(node, content: str, symbols: List[Dict[str, str]], is_default: bool):
    name = _js_field_text(node, "name", content) or "default"
    kind = "export default class" if is_default else "export class"
    symbols.append({"kind": kind, "name": name, "signature": _js_decl(node, content)})
    body = node.child_by_field_name("body")
    if body is None:
        return
    count = 0
    for c in body.children:
        if c.type != "method_definition" or count >= _MAX_METHODS:
            continue
        method = _js_field_text(c, "name", content)
        if not method:
            continue
        symbols.append(
            {
                "kind": "method",
                "name": f"{name}.{method}",
                "signature": _js_decl(c, content),
            }
        )
        count += 1


def _emit_variables(node, content: str, symbols: List[Dict[str, str]]):
    keyword = node.children[0].type if node.children else "const"
    for c in node.children:
        if c.type != "variable_declarator":
            continue
        name = _js_field_text(c, "name", content)
        if not name:
            continue
        sig = _js_truncate(
            _js_collapse(f"{keyword} {content[c.start_byte : c.end_byte]}")
        )
        symbols.append({"kind": f"export {keyword}", "name": name, "signature": sig})


def _emit_commonjs(node, content: str, symbols: List[Dict[str, str]]):
    """Best-effort ``module.exports = { ... }`` / ``exports.x = ...`` capture."""
    if not node.children:
        return
    expr = node.children[0]
    if expr.type != "assignment_expression":
        return
    left = expr.child_by_field_name("left")
    right = expr.child_by_field_name("right")
    if left is None or right is None:
        return
    left_text = _js_collapse(content[left.start_byte : left.end_byte])
    if left.type == "member_expression" and left_text.startswith("exports."):
        # exports.name = <value>
        prop = left_text.split(".", 1)[1]
        symbols.append(
            {
                "kind": "module.exports",
                "name": prop,
                "signature": _js_truncate(left_text),
            }
        )
        return
    if left_text not in ("module.exports", "exports"):
        return
    if right.type == "object":
        _emit_commonjs_object(right, content, symbols)


def _emit_commonjs_object(obj, content: str, symbols: List[Dict[str, str]]):
    count = 0
    for c in obj.children:
        if count >= _MAX_METHODS:
            break
        if c.type == "shorthand_property_identifier":
            name = content[c.start_byte : c.end_byte]
        elif c.type == "pair":
            key = c.child_by_field_name("key")
            name = content[key.start_byte : key.end_byte] if key else ""
        else:
            continue
        if not name:
            continue
        symbols.append(
            {"kind": "module.exports", "name": name, "signature": f"exports.{name}"}
        )
        count += 1
