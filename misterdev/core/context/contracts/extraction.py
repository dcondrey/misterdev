"""Language dispatch for public-symbol extraction."""

from typing import Dict, List

from .python_generic import _extract_generic_symbols, _extract_python_symbols
from .rust_line import _extract_rust_symbols
from .rust_tree_sitter import _extract_rust_symbols_ts


def _extract_public_symbols(content: str, language: str) -> List[Dict[str, str]]:
    """Extract public API symbols from source code.

    Uses line-by-line heuristic parsing (no regex). Works for Rust, Python,
    TypeScript, Go. Not perfect, but catches the signatures that matter for
    cross-task contracts.
    """
    lines = content.splitlines()

    if language in ("rust", "rs"):
        # Prefer tree-sitter (handles multi-line signatures, generics, where
        # clauses); fall back to the line parser when the grammar is absent.
        symbols = _extract_rust_symbols_ts(content)
        if not symbols:
            symbols = _extract_rust_symbols(lines)
    elif language in ("python", "py"):
        symbols = _extract_python_symbols(lines)
    else:
        symbols = _extract_generic_symbols(lines)

    return symbols
