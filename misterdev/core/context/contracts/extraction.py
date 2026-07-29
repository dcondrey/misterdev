"""Language dispatch for public-symbol extraction."""

from typing import Dict, List

from ._log import logger
from .c_tree_sitter import _extract_c_symbols_ts
from .cpp_tree_sitter import _extract_cpp_symbols_ts
from .csharp_tree_sitter import _extract_csharp_symbols_ts
from .javascript_tree_sitter import _extract_javascript_symbols_ts
from .kotlin_tree_sitter import _extract_kotlin_symbols_ts
from .python_generic import _extract_generic_symbols, _extract_python_symbols
from .rust_line import _extract_rust_symbols
from .rust_tree_sitter import _extract_rust_symbols_ts
from .swift_tree_sitter import _extract_swift_symbols_ts
from .typescript_tree_sitter import _extract_typescript_symbols_ts


def _ts_fallback_warn(language: str, content: str) -> None:
    """Warn once when tree-sitter yields nothing for a non-trivial file."""
    if len(content.splitlines()) > 5:
        logger.warning(
            "tree-sitter returned no symbols for %s file (%d lines); "
            "falling back to generic line parser — symbol quality may be degraded.",
            language,
            len(content.splitlines()),
        )


def _extract_public_symbols(content: str, language: str) -> List[Dict[str, str]]:
    """Extract public API symbols from source code.

    Rust, Swift, Kotlin, TypeScript, JavaScript, C#, C, and C++ use tree-sitter
    (multi-line signatures, generics, visibility) and fall back to the line
    heuristic when the grammar is absent or yields nothing; Python and every
    other language use line-based parsing. Catches the signatures that matter
    for cross-task contracts.
    """
    lines = content.splitlines()

    if language in ("rust", "rs"):
        # Prefer tree-sitter (handles multi-line signatures, generics, where
        # clauses); fall back to the line parser when the grammar is absent.
        symbols = _extract_rust_symbols_ts(content)
        if not symbols:
            _ts_fallback_warn(language, content)
            symbols = _extract_rust_symbols(lines)
    elif language in ("python", "py"):
        symbols = _extract_python_symbols(lines)
    elif language in ("swift",):
        symbols = _extract_swift_symbols_ts(content)
    elif language in ("kotlin", "kt"):
        symbols = _extract_kotlin_symbols_ts(content)
        if not symbols:
            _ts_fallback_warn(language, content)
            symbols = _extract_generic_symbols(lines)
    elif language in ("typescript", "ts"):
        symbols = _extract_typescript_symbols_ts(content)
        if not symbols:
            _ts_fallback_warn(language, content)
            symbols = _extract_generic_symbols(lines)
    elif language in ("javascript", "js"):
        symbols = _extract_javascript_symbols_ts(content)
        if not symbols:
            _ts_fallback_warn(language, content)
            symbols = _extract_generic_symbols(lines)
    elif language in ("csharp", "cs"):
        symbols = _extract_csharp_symbols_ts(content)
        if not symbols:
            _ts_fallback_warn(language, content)
            symbols = _extract_generic_symbols(lines)
    elif language in ("cpp", "c++", "cc"):
        symbols = _extract_cpp_symbols_ts(content)
        if not symbols:
            _ts_fallback_warn(language, content)
            symbols = _extract_generic_symbols(lines)
    elif language == "c":
        # Exact match only: "c" must not swallow other language codes.
        symbols = _extract_c_symbols_ts(content)
        if not symbols:
            _ts_fallback_warn(language, content)
            symbols = _extract_generic_symbols(lines)
    else:
        symbols = _extract_generic_symbols(lines)

    return symbols
