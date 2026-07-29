"""Tree-sitter parse-based syntax verification used by the correctness gate."""

from typing import Any

from .parsers import _get_ts_parsers

# Languages whose tree-sitter grammar is trustworthy enough to gate edits on.
# Kotlin is excluded: its grammar emits ERROR nodes on some valid code, which
# would false-reject correct edits. TypeScript is parsed with the TSX grammar
# (a superset) so JSX never trips a false syntax error.
_SYNTAX_CHECK_LANGS = {
    "rust",
    "c",
    "cpp",
    "csharp",
    "swift",
    "kotlin",
    "javascript",
    "typescript",
    "tsx",
}


def _first_error_node(node: Any):
    """Pre-order search for the first ERROR or MISSING node, or None."""
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type == "ERROR" or current.is_missing:
            return current
        stack.extend(reversed(current.children))
    return None


def check_syntax(source: str, lang: str):
    """Verify ``source`` parses cleanly for ``lang`` using tree-sitter.

    Returns ``(ok, message)`` when the language has a trustworthy grammar, or
    ``None`` when it does not (the caller then falls back to a lighter check).
    Unlike brace-counting, this understands strings and comments, so braces in
    a string literal never false-trip and real syntax errors are caught before
    the expensive build gate.
    """
    parsers = _get_ts_parsers()
    key = "tsx" if lang in ("typescript", "tsx") else lang
    if lang not in _SYNTAX_CHECK_LANGS or key not in parsers:
        return None
    tree = parsers[key].parse(source.encode("utf-8"))
    err = _first_error_node(tree.root_node)
    if err is None:
        return True, None
    line = err.start_point[0] + 1
    snippet = source.split("\n")[err.start_point[0]].strip()[:80]
    return False, f"{lang} syntax error near line {line}: {snippet}"
