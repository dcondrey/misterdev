"""Topological context mapping: a tree-sitter symbol graph over the project.

Features:
- Scope-aware symbols via tree-sitter for Python, Rust, TypeScript/TSX,
  JavaScript/JSX, C, C++, C#, Swift, and Kotlin (best-effort).
- Per-file outlines (a symbol table of contents) and a whole-project structural
  map, used to navigate and window large files for focused edits.
- Call-neighbor traversal for related-symbol context, ranked by an optional
  semantic embedder (else lexical).
- check_syntax(): tree-sitter parse-based syntax verification (ERROR/MISSING
  nodes) that understands strings/comments, used by the correctness gate.
- Lazy loading; byte-correct slicing for non-ASCII sources.

This module was split into cohesive sections (parsers, syntax, nodes, cache,
graph, engine); every name below is re-exported so the import path
``my_project_orchestrator.core.context.topography`` is unchanged.
"""

from ._log import logger
from .nodes import SymbolNode, _symbol_to_dict, _symbol_from_dict
from .parsers import _get_ts_parsers, _EXT_TO_LANG, _node_text
from .syntax import _SYNTAX_CHECK_LANGS, _first_error_node, check_syntax
from .cache import _CACHE_FORMAT_VERSION, _TopographyCache
from .graph import _CALL_PATTERN, SymbolGraph
from .engine import TopographyEngine

__all__ = [
    "SymbolNode",
    "SymbolGraph",
    "TopographyEngine",
    "check_syntax",
    "_get_ts_parsers",
    "_TopographyCache",
    "logger",
    "_symbol_to_dict",
    "_symbol_from_dict",
    "_EXT_TO_LANG",
    "_node_text",
    "_SYNTAX_CHECK_LANGS",
    "_first_error_node",
    "_CACHE_FORMAT_VERSION",
    "_CALL_PATTERN",
]
