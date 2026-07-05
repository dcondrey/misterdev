"""Tree-sitter parser loading, language detection, and byte-correct slicing."""

from typing import Dict, Any

from ._log import logger

# Lazy imports for optional heavy dependencies
_ts_parsers: Dict[str, Any] = {}
_ts_available = None


def _get_ts_parsers() -> Dict[str, Any]:
    """Lazy-load tree-sitter parsers for all available language grammars."""
    global _ts_parsers, _ts_available
    if _ts_available is not None:
        return _ts_parsers

    try:
        from tree_sitter import Language, Parser
    except ImportError:
        logger.info("tree-sitter not installed; symbol graph will use basic fallback")
        _ts_available = False
        return _ts_parsers

    grammars = {
        "python": "tree_sitter_python",
        "rust": "tree_sitter_rust",
        "c": "tree_sitter_c",
        "cpp": "tree_sitter_cpp",
        "swift": "tree_sitter_swift",
        "csharp": "tree_sitter_c_sharp",
        "javascript": "tree_sitter_javascript",
        "kotlin": "tree_sitter_kotlin",
    }
    for lang_name, module_name in grammars.items():
        try:
            mod = __import__(module_name)
            parser = Parser(Language(mod.language()))
            _ts_parsers[lang_name] = parser
            logger.debug(f"tree-sitter {lang_name} grammar loaded")
        except ImportError:
            logger.debug(f"tree-sitter grammar not installed: {module_name}")

    # TypeScript ships two grammar entrypoints (.ts and .tsx) rather than the
    # single language() the loop above expects, so load it separately.
    try:
        import tree_sitter_typescript as tsts

        _ts_parsers["typescript"] = Parser(Language(tsts.language_typescript()))
        _ts_parsers["tsx"] = Parser(Language(tsts.language_tsx()))
        logger.debug("tree-sitter typescript grammar loaded")
    except ImportError:
        logger.debug("tree-sitter grammar not installed: tree_sitter_typescript")

    if _ts_parsers:
        _ts_available = True
        logger.info(f"tree-sitter available for: {', '.join(_ts_parsers)}")
    else:
        _ts_available = False
        logger.info("No tree-sitter grammars installed; symbol graph disabled")

    return _ts_parsers


_EXT_TO_LANG: Dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".rs": "rust",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".swift": "swift",
    ".cs": "csharp",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".kt": "kotlin",
    ".kts": "kotlin",
}


def _node_text(src: bytes, node: Any) -> str:
    """Decode a node's span from the UTF-8 source bytes.

    Offsets from tree-sitter are byte positions, so slicing must happen on the
    bytes and decode after, never on the str.
    """
    return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
