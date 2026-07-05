"""The scope-aware tree-sitter symbol graph."""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

from misterdev.utils.file_utils import read_file, is_golden_path

from ._log import logger
from .nodes import SymbolNode
from .parsers import _get_ts_parsers, _EXT_TO_LANG, _node_text
from .cache import _TopographyCache

# Whole-identifier call detection: an identifier directly followed by "(" and
# not preceded by an identifier char, so "reparse(" yields "reparse" — never a
# spurious "parse" — unlike the old `f"{name}(" in content` substring test.
_CALL_PATTERN = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_]\w*)\s*\(")


class SymbolGraph:
    """Robust dependency graph with scope-aware parsing."""

    def __init__(self, project_path: Path, golden_paths=None):
        self.project_path = project_path
        self.symbols: Dict[str, SymbolNode] = {}
        self.parsers = _get_ts_parsers()
        self.golden_paths = golden_paths or []
        self.cache_path = Path(project_path) / ".orchestrator" / "topography_cache.json"

    def build(self):
        logger.info(f"Building symbol graph for {self.project_path}")
        self.symbols.clear()

        if not self.parsers:
            logger.info("Tree-sitter not available; symbol graph disabled")
            return

        # Skip dependency/build-output dirs across toolchains so the symbol map
        # reflects real source, not generated artifacts (e.g. Swift's `.build`
        # derived sources and headers, Xcode `DerivedData`, CocoaPods, wasm-pack
        # `pkg`). Without this a frontend task gets generated junk as context.
        _skip = frozenset(
            (
                ".venv",
                ".git",
                "__pycache__",
                "node_modules",
                "target",
                "build",
                ".build",
                "dist",
                "pkg",
                "Pods",
                "DerivedData",
                ".gradle",
                ".next",
                "vendor",
            )
        )
        supported_exts = {
            ext for ext, lang in _EXT_TO_LANG.items() if lang in self.parsers
        }

        # Walk with in-place directory pruning so we never DESCEND into skipped
        # or hidden dirs (node_modules, .git, .claude, …). rglob would walk all
        # of node_modules before filtering — slow on a real frontend repo — and a
        # large dot-dir like .claude could otherwise crowd real source out of the
        # outline's file cap, leaving the decomposer with no code to ground on.
        source_files: List[Path] = []
        for dirpath, dirnames, filenames in os.walk(self.project_path):
            dirnames[:] = [
                d for d in dirnames if d not in _skip and not d.startswith(".")
            ]
            for fn in filenames:
                f = Path(dirpath) / fn
                if f.suffix not in supported_exts:
                    continue
                if is_golden_path(
                    str(f.relative_to(self.project_path)), self.golden_paths
                ):
                    continue
                source_files.append(f)

        if not source_files:
            logger.info("No supported source files found; symbol graph will be empty")
            return

        cache = _TopographyCache(self.cache_path)
        cache.load()
        live_paths: Set[str] = set()

        for src_file in source_files:
            lang = _EXT_TO_LANG.get(src_file.suffix)
            if not (lang and lang in self.parsers):
                continue
            rel_path = str(src_file.relative_to(self.project_path))
            live_paths.add(rel_path)

            # Content-hash key: an edit changes the bytes and so the key, which
            # is the cache-invalidation signal (mtime is never trusted).
            content_bytes = self._read_bytes(src_file)
            key = (
                _TopographyCache.make_key(content_bytes, lang)
                if content_bytes is not None
                else None
            )

            cached = cache.get(rel_path, key) if key is not None else None
            if cached is not None:
                for sym in cached:
                    self.symbols[f"{sym.file_path}:{sym.name}"] = sym
                continue

            file_symbols = self._parse_file(src_file, lang, content_bytes=content_bytes)
            for sym in file_symbols:
                self.symbols[f"{sym.file_path}:{sym.name}"] = sym
            # Only cache a clean parse: a read failure yields no key, so we never
            # persist a wrong (empty) result under a real file's slot.
            if key is not None:
                cache.put(rel_path, key, file_symbols)

        cache.prune(live_paths)
        cache.save()

        self._resolve_references()
        logger.info(f"Symbol graph complete: {len(self.symbols)} symbols.")

    def _read_bytes(self, file_path: Path) -> Optional[bytes]:
        """Read a file's UTF-8 bytes, or None on any I/O/decode error.

        A None result forces a (failed) parse path with no caching, so a
        transiently unreadable file never poisons the cache with empty symbols.
        """
        try:
            return read_file(file_path).encode("utf-8")
        except (OSError, UnicodeError) as e:
            logger.debug(f"Topography could not read {file_path}: {e}")
            return None

    def _parse_file(
        self, file_path: Path, lang: str, content_bytes: Optional[bytes] = None
    ) -> List[SymbolNode]:
        # tree-sitter reports BYTE offsets; slice the UTF-8 bytes (not the str)
        # so non-ASCII content before a symbol doesn't shift and mangle names.
        content = (
            content_bytes if content_bytes is not None else self._read_bytes(file_path)
        )
        if content is None:
            return []
        parser = self.parsers[lang]
        tree = parser.parse(content)
        rel_path = str(file_path.relative_to(self.project_path))

        # Parse into a private dict so the symbols for THIS file are isolated
        # (for caching) without disturbing the graph-wide self.symbols.
        prev_symbols = self.symbols
        self.symbols = {}
        try:
            self._dispatch_traverse(tree.root_node, content, rel_path, lang)
            parsed = list(self.symbols.values())
        finally:
            self.symbols = prev_symbols
        return parsed

    def _dispatch_traverse(self, root: Any, content: bytes, rel_path: str, lang: str):
        if lang == "rust":
            self._traverse_rust(root, content, rel_path)
        elif lang in ("typescript", "tsx", "javascript"):
            self._traverse_typescript(root, content, rel_path)
        elif lang == "kotlin":
            self._traverse_kotlin(root, content, rel_path)
        elif lang in ("c", "cpp"):
            self._traverse_clike(root, content, rel_path)
        elif lang == "swift":
            self._traverse_swift(root, content, rel_path)
        elif lang == "csharp":
            self._traverse_csharp(root, content, rel_path)
        else:
            self._traverse_python(root, content, rel_path)

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
                full = f"{parent_class}.{name}" if parent_class else name
                self._add_symbol(
                    full,
                    file_path,
                    "method" if parent_class else "function",
                    node,
                    content,
                )

        elif t in ("class_declaration", "interface_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node:
                name = _node_text(content, name_node)
                kind = "interface" if t == "interface_declaration" else "class"
                self._add_symbol(name, file_path, kind, node, content)
                body_node = node.child_by_field_name("body")
                if body_node:
                    for child in body_node.children:
                        self._traverse_typescript(
                            child, content, file_path, parent_class=name
                        )
                return

        elif t == "enum_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                self._add_symbol(
                    _node_text(content, name_node), file_path, "enum", node, content
                )

        elif t in ("lexical_declaration", "variable_declaration"):
            # const/let X = () => {...} or function(){...}: how React components
            # and most modern TS/JS functions are written. Capture the binding.
            for decl in node.children:
                if decl.type != "variable_declarator":
                    continue
                value = decl.child_by_field_name("value")
                name_node = decl.child_by_field_name("name")
                if (
                    name_node
                    and value is not None
                    and value.type in ("arrow_function", "function_expression")
                ):
                    name = _node_text(content, name_node)
                    full = f"{parent_class}.{name}" if parent_class else name
                    self._add_symbol(
                        full,
                        file_path,
                        "method" if parent_class else "function",
                        node,
                        content,
                    )

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
                self._add_symbol(name, file_path, "type", node, content)

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
                kind = (
                    "class"
                    if t == "class_specifier"
                    else ("enum" if t == "enum_specifier" else "struct")
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
                    "protocol"
                    if t == "protocol_declaration"
                    else "struct"
                    if "struct" in kinds
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

    def _traverse_kotlin(
        self, node: Any, content: str, file_path: str, parent: Optional[str] = None
    ):
        # Best-effort: the Kotlin grammar reports ERROR nodes on some valid
        # code, so always recurse all children (declarations under an ERROR
        # node are still captured) and never use Kotlin for the syntax gate.
        t = node.type
        if t == "function_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = _node_text(content, name_node)
                full = f"{parent}.{name}" if parent else name
                self._add_symbol(
                    full, file_path, "method" if parent else "function", node, content
                )
        elif t in ("class_declaration", "object_declaration"):
            name_node = node.child_by_field_name("name") or self._first_child_of_type(
                node, "type_identifier"
            )
            if name_node:
                name = _node_text(content, name_node)
                kind = "object" if t == "object_declaration" else "class"
                self._add_symbol(name, file_path, kind, node, content)
                for child in node.children:
                    self._traverse_kotlin(child, content, file_path, parent=name)
                return

        for child in node.children:
            self._traverse_kotlin(child, content, file_path, parent)

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
        # Names can collide across files (two functions both named ``run``).
        # Keying by name alone collapsed them, so every ``run()`` call resolved to
        # one arbitrary definition — a misattributed edge. Resolve scope-aware
        # instead: prefer a definition in the caller's OWN file, then a
        # globally-unique definition; when several files define the name and none
        # is local, the call is genuinely ambiguous without import resolution, so
        # we add no edge rather than guess wrong.
        name_to_keys: Dict[str, List[str]] = {}
        for key, s in self.symbols.items():
            name_to_keys.setdefault(s.name, []).append(key)
        for key, symbol in self.symbols.items():
            called = {m.group(1) for m in _CALL_PATTERN.finditer(symbol.content)}
            for name in called:
                candidates = name_to_keys.get(name)
                if not candidates:
                    continue
                same_file = [
                    k
                    for k in candidates
                    if k != key and self.symbols[k].file_path == symbol.file_path
                ]
                if same_file:
                    targets = same_file
                elif len(candidates) == 1 and candidates[0] != key:
                    targets = candidates
                else:
                    continue
                for other_key in targets:
                    symbol.outgoing_calls.add(other_key)
                    self.symbols[other_key].incoming_calls.add(key)

    def file_symbols(self, file_path: str) -> List[SymbolNode]:
        """Symbols defined in one file, ordered by start line."""
        return sorted(
            (s for s in self.symbols.values() if s.file_path == file_path),
            key=lambda s: s.start_line,
        )

    def _match_files(self, file_path: str) -> Set[str]:
        """Graph file paths an error path refers to: an exact match when present,
        else a UNIQUE suffix match so a path reported relative to a sub-target's
        cwd (``src/app.ts``) still resolves against the root-relative key
        (``frontend/src/app.ts``). An ambiguous suffix yields no match rather
        than a wrong one."""
        exact = {s.file_path for s in self.symbols.values() if s.file_path == file_path}
        if exact:
            return exact
        suffix = "/" + file_path
        matches = {
            s.file_path for s in self.symbols.values() if s.file_path.endswith(suffix)
        }
        return matches if len(matches) == 1 else set()

    def symbol_at_line(self, file_path: str, line: int) -> Optional[str]:
        """Key of the narrowest symbol enclosing a 1-indexed source line, or None.

        ``start_line``/``end_line`` are 0-indexed tree-sitter rows (rendered
        ``+1`` for humans), so a 1-indexed error line maps to row ``line - 1``.
        The narrowest enclosing span wins so a method attributes to itself, not
        its containing class.
        """
        files = self._match_files(file_path)
        if not files:
            return None
        row = line - 1
        best_key: Optional[str] = None
        best_span: Optional[int] = None
        for key, sym in self.symbols.items():
            if sym.file_path not in files:
                continue
            if sym.start_line <= row <= sym.end_line:
                span = sym.end_line - sym.start_line
                if best_span is None or span < best_span:
                    best_span = span
                    best_key = key
        return best_key

    def callers_of(self, key: str) -> List[str]:
        """Names of the symbols that call the symbol identified by ``key``
        (capped). Keyed by the unique ``file_path:name`` id, so same-named
        symbols in other files are never conflated."""
        sym = self.symbols.get(key)
        if sym is None:
            return []
        callers: List[str] = []
        for caller_key in sym.incoming_calls:
            caller = self.symbols.get(caller_key)
            if caller and caller.name not in callers:
                callers.append(caller.name)
        return callers[:5]

    def file_outline(self, file_path: str) -> str:
        """A compact table of contents for one file: each symbol with its lines.

        Lets the model navigate a large file it is editing and place precise
        SEARCH anchors without scanning the whole body line by line.
        """
        return "\n".join(
            f"  L{s.start_line + 1}-{s.end_line + 1}: {s.kind} {s.name}"
            for s in self.file_symbols(file_path)
        )

    def project_outline(self, max_files: int = 300, max_syms_per_file: int = 60) -> str:
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
