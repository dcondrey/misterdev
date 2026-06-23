"""High-Rigor Topological Context Mapping Engine.

Features:
- Scope-aware symbol resolution via Tree-Sitter.
- LanceDB-backed semantic vector indexing.
- Multi-degree graph traversal for 'Omniscient Context'.
- Lazy loading for high-speed CLI performance.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Any

from my_project_orchestrator.logging_setup import setup_logger
from my_project_orchestrator.utils.file_utils import read_file, is_golden_path

logger = setup_logger(__name__)

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


class SymbolNode:
    def __init__(
        self,
        name: str,
        file_path: str,
        kind: str,
        start_line: int,
        end_line: int,
        content: str,
    ):
        self.name = name
        self.file_path = file_path
        self.kind = kind  # 'function', 'class', 'method'
        self.start_line = start_line
        self.end_line = end_line
        self.content = content
        self.outgoing_calls: Set[str] = set()
        self.incoming_calls: Set[str] = set()
        self.imports: List[Dict[str, str]] = []  # {name: ..., module: ...}

    def __repr__(self):
        return f"Symbol({self.kind}:{self.name} in {self.file_path})"


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
}


def _node_text(src: bytes, node: Any) -> str:
    """Decode a node's span from the UTF-8 source bytes.

    Offsets from tree-sitter are byte positions, so slicing must happen on the
    bytes and decode after, never on the str.
    """
    return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


class SymbolGraph:
    """Robust dependency graph with scope-aware parsing."""

    def __init__(self, project_path: Path, golden_paths=None):
        self.project_path = project_path
        self.symbols: Dict[str, SymbolNode] = {}
        self.parsers = _get_ts_parsers()
        self.golden_paths = golden_paths or []

    def build(self):
        logger.info(f"Building symbol graph for {self.project_path}")
        self.symbols.clear()

        if not self.parsers:
            logger.info("Tree-sitter not available; symbol graph disabled")
            return

        _skip = frozenset(
            (".venv", ".git", "__pycache__", "node_modules", "target", "build", "dist")
        )
        supported_exts = {
            ext for ext, lang in _EXT_TO_LANG.items() if lang in self.parsers
        }
        source_files = [
            f
            for f in self.project_path.rglob("*")
            if f.suffix in supported_exts
            and not (_skip & set(f.parts))
            and not is_golden_path(
                str(f.relative_to(self.project_path)), self.golden_paths
            )
        ]

        if not source_files:
            logger.info("No supported source files found; symbol graph will be empty")
            return

        for src_file in source_files:
            lang = _EXT_TO_LANG.get(src_file.suffix)
            if lang and lang in self.parsers:
                self._parse_file(src_file, lang)

        self._resolve_references()
        logger.info(f"Symbol graph complete: {len(self.symbols)} symbols.")

    def _parse_file(self, file_path: Path, lang: str):
        # tree-sitter reports BYTE offsets; slice the UTF-8 bytes (not the str)
        # so non-ASCII content before a symbol doesn't shift and mangle names.
        content = read_file(file_path).encode("utf-8")
        parser = self.parsers[lang]
        tree = parser.parse(content)
        rel_path = str(file_path.relative_to(self.project_path))

        if lang == "rust":
            self._traverse_rust(tree.root_node, content, rel_path)
        elif lang in ("typescript", "tsx"):
            self._traverse_typescript(tree.root_node, content, rel_path)
        elif lang in ("c", "cpp"):
            self._traverse_clike(tree.root_node, content, rel_path)
        elif lang == "swift":
            self._traverse_swift(tree.root_node, content, rel_path)
        elif lang == "csharp":
            self._traverse_csharp(tree.root_node, content, rel_path)
        else:
            self._traverse_python(tree.root_node, content, rel_path)

    def _traverse_python(
        self,
        node: Any,
        content: str,
        file_path: str,
        parent_class: Optional[str] = None,
    ):
        if node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = _node_text(content, name_node)
                full_name = f"{parent_class}.{name}" if parent_class else name
                self._add_symbol(
                    full_name,
                    file_path,
                    "method" if parent_class else "function",
                    node,
                    content,
                )

        elif node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = _node_text(content, name_node)
                self._add_symbol(name, file_path, "class", node, content)
                body_node = node.child_by_field_name("body")
                if body_node:
                    for child in body_node.children:
                        self._traverse_python(
                            child, content, file_path, parent_class=name
                        )
                return

        for child in node.children:
            self._traverse_python(child, content, file_path, parent_class)

    def _traverse_rust(
        self, node: Any, content: str, file_path: str, parent: Optional[str] = None
    ):
        if node.type == "function_item":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = _node_text(content, name_node)
                full_name = f"{parent}::{name}" if parent else name
                kind = "method" if parent else "function"
                self._add_symbol(full_name, file_path, kind, node, content)

        elif node.type == "struct_item":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = _node_text(content, name_node)
                self._add_symbol(name, file_path, "struct", node, content)

        elif node.type == "enum_item":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = _node_text(content, name_node)
                self._add_symbol(name, file_path, "enum", node, content)

        elif node.type == "trait_item":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = _node_text(content, name_node)
                self._add_symbol(name, file_path, "trait", node, content)

        elif node.type == "impl_item":
            type_node = node.child_by_field_name("type")
            if type_node:
                impl_name = _node_text(content, type_node)
                body_node = node.child_by_field_name("body")
                if body_node:
                    for child in body_node.children:
                        self._traverse_rust(child, content, file_path, parent=impl_name)
                return

        for child in node.children:
            self._traverse_rust(child, content, file_path, parent)

    def _traverse_typescript(
        self,
        node: Any,
        content: str,
        file_path: str,
        parent_class: Optional[str] = None,
    ):
        t = node.type
        if t == "function_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = _node_text(content, name_node)
                self._add_symbol(name, file_path, "function", node, content)

        elif t in ("class_declaration", "interface_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node:
                name = _node_text(content, name_node)
                self._add_symbol(name, file_path, "class", node, content)
                body_node = node.child_by_field_name("body")
                if body_node:
                    for child in body_node.children:
                        self._traverse_typescript(
                            child, content, file_path, parent_class=name
                        )
                return

        elif t == "method_definition":
            name_node = node.child_by_field_name("name")
            if name_node and parent_class:
                name = _node_text(content, name_node)
                self._add_symbol(
                    f"{parent_class}.{name}", file_path, "method", node, content
                )

        elif t == "type_alias_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = _node_text(content, name_node)
                self._add_symbol(name, file_path, "class", node, content)

        elif t == "export_statement":
            for child in node.children:
                self._traverse_typescript(child, content, file_path, parent_class)
            return

        for child in node.children:
            self._traverse_typescript(child, content, file_path, parent_class)

    def _traverse_clike(
        self, node: Any, content: str, file_path: str, parent: Optional[str] = None
    ):
        t = node.type
        if t == "function_definition":
            decl = node.child_by_field_name("declarator")
            name = self._descend_identifier(decl, content) if decl else None
            if name:
                full = f"{parent}::{name}" if parent else name
                self._add_symbol(
                    full, file_path, "method" if parent else "function", node, content
                )
        elif t in ("struct_specifier", "class_specifier", "enum_specifier"):
            name_node = node.child_by_field_name("name")
            if name_node:
                name = _node_text(content, name_node)
                kind = "class" if t == "class_specifier" else (
                    "enum" if t == "enum_specifier" else "struct"
                )
                self._add_symbol(name, file_path, kind, node, content)
                body = node.child_by_field_name("body")
                if body:
                    for child in body.children:
                        self._traverse_clike(child, content, file_path, parent=name)
                return
        elif t == "type_definition":
            name_node = self._last_child_of_type(node, "type_identifier")
            if name_node:
                name = _node_text(content, name_node)
                self._add_symbol(name, file_path, "type", node, content)
        elif t == "namespace_definition":
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    self._traverse_clike(child, content, file_path, parent)
                return

        for child in node.children:
            self._traverse_clike(child, content, file_path, parent)

    def _traverse_swift(
        self, node: Any, content: str, file_path: str, parent: Optional[str] = None
    ):
        t = node.type
        if t == "function_declaration":
            name = self._first_child_text(node, content, ("simple_identifier",))
            if name:
                full = f"{parent}.{name}" if parent else name
                self._add_symbol(
                    full, file_path, "method" if parent else "function", node, content
                )
        elif t in ("class_declaration", "protocol_declaration"):
            name = self._first_child_text(node, content, ("type_identifier",))
            if name:
                # The Swift grammar models struct as a class_declaration with a
                # `struct` keyword child; distinguish so the kind is accurate.
                kinds = {c.type for c in node.children}
                kind = (
                    "protocol" if t == "protocol_declaration"
                    else "struct" if "struct" in kinds
                    else "class"
                )
                self._add_symbol(name, file_path, kind, node, content)
                for child in node.children:
                    if child.type in ("class_body", "protocol_body"):
                        for member in child.children:
                            self._traverse_swift(
                                member, content, file_path, parent=name
                            )
                return

        for child in node.children:
            self._traverse_swift(child, content, file_path, parent)

    _CSHARP_TYPE_KINDS = {
        "class_declaration": "class",
        "struct_declaration": "struct",
        "interface_declaration": "interface",
        "enum_declaration": "enum",
        "record_declaration": "record",
    }

    def _traverse_csharp(
        self, node: Any, content: str, file_path: str, parent: Optional[str] = None
    ):
        t = node.type
        if t in self._CSHARP_TYPE_KINDS:
            name_node = node.child_by_field_name("name")
            if name_node:
                name = _node_text(content, name_node)
                self._add_symbol(
                    name, file_path, self._CSHARP_TYPE_KINDS[t], node, content
                )
                body = node.child_by_field_name("body")
                if body is None:
                    body = self._first_child_of_type(node, "declaration_list")
                if body:
                    for child in body.children:
                        self._traverse_csharp(child, content, file_path, parent=name)
                return
        elif t in ("method_declaration", "constructor_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node:
                name = _node_text(content, name_node)
                full = f"{parent}.{name}" if parent else name
                self._add_symbol(
                    full, file_path, "method" if parent else "function", node, content
                )
        elif t == "property_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = _node_text(content, name_node)
                full = f"{parent}.{name}" if parent else name
                self._add_symbol(full, file_path, "property", node, content)

        for child in node.children:
            self._traverse_csharp(child, content, file_path, parent)

    @staticmethod
    def _first_child_of_type(node: Any, type_name: str) -> Any:
        for c in node.children:
            if c.type == type_name:
                return c
        return None

    @staticmethod
    def _descend_identifier(node: Any, content: str) -> Optional[str]:
        """Breadth-first search for the first declarator identifier.

        A C/C++ function name is wrapped in declarator nodes (pointer, array,
        parenthesized); the shallowest ``identifier``/``field_identifier`` on
        the name path is the symbol name. BFS finds it before parameter names.
        """
        queue = [node]
        while queue:
            n = queue.pop(0)
            if n.type in ("identifier", "field_identifier"):
                return _node_text(content, n)
            queue.extend(n.children)
        return None

    @staticmethod
    def _first_child_text(node: Any, content: str, types: tuple) -> Optional[str]:
        for c in node.children:
            if c.type in types:
                return _node_text(content, c)
        return None

    @staticmethod
    def _last_child_of_type(node: Any, type_name: str) -> Any:
        found = None
        for c in node.children:
            if c.type == type_name:
                found = c
        return found

    def _add_symbol(
        self, name: str, file_path: str, kind: str, node: Any, content: str
    ):
        key = f"{file_path}:{name}"
        symbol_content = _node_text(content, node)
        self.symbols[key] = SymbolNode(
            name,
            file_path,
            kind,
            node.start_point.row,
            node.end_point.row,
            symbol_content,
        )

    def _resolve_references(self):
        name_to_key = {s.name: key for key, s in self.symbols.items()}
        call_patterns = {name: f"{name}(" for name in name_to_key}
        for key, symbol in self.symbols.items():
            for name, pattern in call_patterns.items():
                other_key = name_to_key[name]
                if other_key != key and pattern in symbol.content:
                    symbol.outgoing_calls.add(other_key)
                    self.symbols[other_key].incoming_calls.add(key)

    def file_symbols(self, file_path: str) -> List[SymbolNode]:
        """Symbols defined in one file, ordered by start line."""
        return sorted(
            (s for s in self.symbols.values() if s.file_path == file_path),
            key=lambda s: s.start_line,
        )

    def file_outline(self, file_path: str) -> str:
        """A compact table of contents for one file: each symbol with its lines.

        Lets the model navigate a large file it is editing and place precise
        SEARCH anchors without scanning the whole body line by line.
        """
        return "\n".join(
            f"  L{s.start_line + 1}-{s.end_line + 1}: {s.kind} {s.name}"
            for s in self.file_symbols(file_path)
        )

    def project_outline(
        self, max_files: int = 300, max_syms_per_file: int = 60
    ) -> str:
        """A whole-project structural map: every file with its top-level symbols.

        Far denser than reading file heads — it conveys the architecture (what
        exists, where) within a small token budget so planning and editing are
        grounded in the entire project, not a first-N-lines slice.
        """
        by_file: Dict[str, List[SymbolNode]] = {}
        for s in self.symbols.values():
            by_file.setdefault(s.file_path, []).append(s)
        if not by_file:
            return ""
        out = []
        for path in sorted(by_file)[:max_files]:
            syms = sorted(by_file[path], key=lambda s: s.start_line)
            shown = ", ".join(f"{s.kind} {s.name}" for s in syms[:max_syms_per_file])
            if len(syms) > max_syms_per_file:
                shown += f", +{len(syms) - max_syms_per_file} more"
            out.append(f"{path}: {shown}")
        if len(by_file) > max_files:
            out.append(f"(... {len(by_file) - max_files} more files)")
        return "\n".join(out)


class TopographyEngine:
    """Topography Engine with Vector Persistence and Lazy Loading."""

    def __init__(self, project_path: Path, llm_client: Any, golden_paths=None):
        self.project_path = project_path
        self.llm = llm_client
        self.graph = SymbolGraph(project_path, golden_paths=golden_paths)
        self._initialized = False

    def initialize(self, force: bool = False):
        if self._initialized and not force:
            return

        logger.info("Initializing Topography Engine...")
        self.graph.build()
        logger.info(f"Symbol graph: {len(self.graph.symbols)} symbols indexed")
        self._initialized = True

    def get_file_outline(self, file_path: str) -> str:
        self.initialize()
        return self.graph.file_outline(file_path)

    def get_file_symbols(self, file_path: str):
        self.initialize()
        return self.graph.file_symbols(file_path)

    def get_project_outline(self) -> str:
        self.initialize()
        return self.graph.project_outline()

    def get_context_for_task(
        self, query: str, related_files: List[str], max_symbols: int = 30, ranker=None
    ) -> str:
        """Retrieves functional neighborhood and semantic context. Triggers lazy init.

        When more candidate symbols are found than fit (``max_symbols``) and a
        semantic ``ranker`` is supplied, the kept symbols are the ones most
        relevant to ``query`` rather than an arbitrary slice.
        """
        self.initialize()

        context_symbols: Set[str] = set()
        for file in related_files:
            for key, sym in self.graph.symbols.items():
                if sym.file_path == file:
                    context_symbols.add(key)
                    context_symbols.update(sym.outgoing_calls)
                    context_symbols.update(sym.incoming_calls)

        if not context_symbols:
            return ""

        # Cap to avoid blowing LLM context, keeping the most task-relevant
        # symbols when a semantic ranker is available.
        if ranker is not None and len(context_symbols) > max_symbols:
            candidates = {
                key: self.graph.symbols[key].content
                for key in context_symbols
                if key in self.graph.symbols
            }
            symbol_list = ranker.top_k(query, candidates, max_symbols)
        else:
            symbol_list = list(context_symbols)[:max_symbols]

        output = "## Topological Context\n"
        by_file: Dict[str, List[SymbolNode]] = {}
        for key in symbol_list:
            if key in self.graph.symbols:
                sym = self.graph.symbols[key]
                by_file.setdefault(sym.file_path, []).append(sym)

        for file_path, syms in by_file.items():
            output += f"\n### Symbols in {file_path}\n"
            for sym in syms:
                output += f"\n# {sym.kind.upper()}: {sym.name}\n{sym.content}\n"

        if len(context_symbols) > max_symbols:
            output += (
                f"\n(... {len(context_symbols) - max_symbols} more symbols omitted)\n"
            )

        return output
