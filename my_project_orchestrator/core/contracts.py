"""Cross-task interface contract registry.

After each task completes, extracts the public API it created or modified.
Before executing downstream tasks, injects those contracts into prompts
so the LLM knows the exact signatures it must honor.

This addresses ~30% of multi-task build failures where one task assumes
a different interface than what the previous task actually created.
"""

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional

from my_project_orchestrator.logging_setup import setup_logger
from my_project_orchestrator.utils.file_utils import atomic_write

logger = setup_logger(__name__)


class Contract:
    """A public API contract extracted from a completed task."""

    def __init__(self, task_id: str, file_path: str, symbols: List[Dict[str, str]]):
        self.task_id = task_id
        self.file_path = file_path
        self.symbols = symbols  # [{name, kind, signature}]

    def format_for_prompt(self) -> str:
        lines = [f"### {self.file_path} (from {self.task_id})"]
        for sym in self.symbols:
            kind = sym.get("kind", "symbol")
            name = sym.get("name", "?")
            sig = sym.get("signature", "")
            lines.append(f"- {kind}: `{sig or name}`")
        return "\n".join(lines)


class ContractRegistry:
    """Manages interface contracts across tasks.

    After a task completes, call `extract_contracts()` to record what it exported.
    Before a task executes, call `get_contracts_for_task()` to get the interfaces
    it depends on.
    """

    def __init__(self, project_path: Path):
        self.contracts: Dict[str, List[Contract]] = {}  # task_id -> contracts
        self._file = project_path / ".orchestrator" / "contracts.json"
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if self._file.exists():
            try:
                data = json.loads(self._file.read_text(encoding="utf-8"))
                for task_id, entries in data.items():
                    self.contracts[task_id] = [
                        Contract(task_id, e["file_path"], e["symbols"])
                        for e in entries
                    ]
            except (json.JSONDecodeError, OSError, KeyError):
                self.contracts = {}

    def _save(self):
        data = {}
        for task_id, contracts in self.contracts.items():
            data[task_id] = [
                {"file_path": c.file_path, "symbols": c.symbols}
                for c in contracts
            ]
        atomic_write(self._file, json.dumps(data, indent=2))

    def extract_contracts(
        self, task_id: str, modified_files: List[str],
        project_path: Path, llm_client, language: str = "rust",
    ) -> List[Contract]:
        """Extract public API from files modified by a completed task."""
        contracts = []
        for file_path in modified_files:
            full_path = project_path / file_path
            if not full_path.exists():
                continue
            try:
                content = full_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            if len(content.strip()) == 0:
                continue

            symbols = _extract_public_symbols(content, language)
            if symbols:
                contracts.append(Contract(task_id, file_path, symbols))

        with self._lock:
            self.contracts[task_id] = contracts
            self._save()
        logger.info(f"Extracted {sum(len(c.symbols) for c in contracts)} contracts from {task_id}")
        return contracts

    def get_contracts_for_task(self, dependency_ids: List[str]) -> str:
        """Format contracts from dependency tasks as prompt context."""
        if not dependency_ids:
            return ""

        relevant = []
        with self._lock:
            for dep_id in dependency_ids:
                if dep_id in self.contracts:
                    relevant.extend(self.contracts[dep_id])

        if not relevant:
            return ""

        lines = ["## Interface Contracts (from completed dependency tasks)",
                 "Your code MUST use these exact signatures. Do not guess or assume different names.\n"]
        for contract in relevant:
            lines.append(contract.format_for_prompt())
        return "\n".join(lines)

    def get_all_contracts_summary(self) -> str:
        """Summary for reporting."""
        total = sum(len(cs) for cs in self.contracts.values())
        return f"{len(self.contracts)} tasks, {total} total contracts"


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


def _extract_rust_symbols_ts(content: str) -> List[Dict[str, str]]:
    """Extract public Rust API using tree-sitter, reusing topography's parser.

    Returns [] when tree-sitter or the Rust grammar is unavailable so the
    caller can fall back to the line-based parser.
    """
    try:
        from my_project_orchestrator.core.topography import _get_ts_parsers
        parser = _get_ts_parsers().get("rust")
    except Exception:
        return []
    if parser is None:
        return []
    try:
        tree = parser.parse(bytes(content, "utf8"))
    except Exception:
        return []
    symbols: List[Dict[str, str]] = []
    _walk_rust_ts(tree.root_node, content, symbols, parent=None)
    return symbols


def _ts_is_pub(node) -> bool:
    return any(c.type == "visibility_modifier" for c in node.children)


def _ts_field_text(node, field: str, content: str) -> str:
    n = node.child_by_field_name(field)
    return content[n.start_byte:n.end_byte] if n else ""


def _ts_decl(node, content: str) -> str:
    """Declaration text up to the body (captures generics/where, drops body)."""
    body = node.child_by_field_name("body")
    end = body.start_byte if body else node.end_byte
    return " ".join(content[node.start_byte:end].split()).strip()


def _walk_rust_ts(node, content: str, symbols: List[Dict[str, str]], parent: Optional[str]):
    t = node.type
    if t == "function_item":
        if _ts_is_pub(node):
            name = _ts_field_text(node, "name", content)
            full = f"{parent}::{name}" if parent else name
            symbols.append({"kind": "pub fn", "name": full, "signature": _ts_decl(node, content)})
        return
    if t == "struct_item" and _ts_is_pub(node):
        name = _ts_field_text(node, "name", content)
        sig = _ts_decl(node, content)
        fields = _ts_pub_members(node, "field_declaration", content)
        if fields:
            sig += " { " + ", ".join(fields) + " }"
        symbols.append({"kind": "pub struct", "name": name, "signature": sig})
        return
    if t == "enum_item" and _ts_is_pub(node):
        name = _ts_field_text(node, "name", content)
        variants = _ts_variant_names(node, content)
        sig = _ts_decl(node, content)
        if variants:
            sig += " { " + ", ".join(variants) + " }"
        symbols.append({"kind": "pub enum", "name": name, "signature": sig})
        return
    if t == "trait_item" and _ts_is_pub(node):
        name = _ts_field_text(node, "name", content)
        methods = _ts_trait_methods(node, content)
        sig = _ts_decl(node, content)
        if methods:
            sig += " { " + "; ".join(methods) + "; }"
        symbols.append({"kind": "pub trait", "name": name, "signature": sig})
        return
    if t == "type_item" and _ts_is_pub(node):
        name = _ts_field_text(node, "name", content)
        text = " ".join(content[node.start_byte:node.end_byte].split()).rstrip(";")
        symbols.append({"kind": "pub type", "name": name, "signature": text})
        return
    if t == "impl_item":
        type_node = node.child_by_field_name("type")
        body = node.child_by_field_name("body")
        if type_node is not None and body is not None:
            impl_name = content[type_node.start_byte:type_node.end_byte].split("<")[0].strip()
            for c in body.children:
                _walk_rust_ts(c, content, symbols, parent=impl_name)
        return
    for c in node.children:
        _walk_rust_ts(c, content, symbols, parent)


def _ts_pub_members(node, child_type: str, content: str) -> List[str]:
    body = node.child_by_field_name("body")
    members = []
    if body is not None:
        for c in body.children:
            if c.type == child_type and any(g.type == "visibility_modifier" for g in c.children):
                members.append(" ".join(content[c.start_byte:c.end_byte].split()).rstrip(","))
    return members[:10]


def _ts_variant_names(node, content: str) -> List[str]:
    body = node.child_by_field_name("body")
    names = []
    if body is not None:
        for c in body.children:
            if c.type == "enum_variant":
                n = c.child_by_field_name("name")
                if n is not None:
                    names.append(content[n.start_byte:n.end_byte])
    return names[:15]


def _ts_trait_methods(node, content: str) -> List[str]:
    body = node.child_by_field_name("body")
    methods = []
    if body is not None:
        for c in body.children:
            if c.type in ("function_signature_item", "function_item"):
                b = c.child_by_field_name("body")
                end = b.start_byte if b is not None else c.end_byte
                methods.append(" ".join(content[c.start_byte:end].split()).rstrip(";"))
    return methods[:10]


def _extract_rust_symbols(lines: List[str]) -> List[Dict[str, str]]:
    """Extract public API from Rust source.

    Handles: pub fn, pub(crate) fn, pub struct (with fields), pub enum,
    pub trait (with method signatures), pub type, pub const, impl blocks.
    Collects multi-line signatures up to the opening brace.
    """
    symbols = []
    i = 0
    brace_depth = 0
    while i < len(lines):
        stripped = lines[i].strip()
        line_opens = stripped.count("{") - stripped.count("}")

        # Skip lines inside bodies (impl, struct, enum, trait blocks)
        if brace_depth > 0:
            brace_depth += line_opens
            i += 1
            continue

        # Match pub, pub(crate), pub(super)
        if not (stripped.startswith("pub ") or stripped.startswith("pub(")):
            # Check for impl blocks (extract public methods inside)
            if stripped.startswith("impl ") or stripped.startswith("impl<"):
                impl_name = _extract_impl_name(stripped)
                impl_methods = _extract_impl_methods(lines, i)
                for m in impl_methods:
                    m["name"] = f"{impl_name}::{m['name']}"
                symbols.extend(impl_methods)
                # Skip past the impl block
                brace_depth += line_opens
            else:
                brace_depth += line_opens
            i += 1
            continue

        # Strip visibility qualifier to get the declaration
        after_vis = _strip_visibility(stripped)

        if after_vis.startswith("fn "):
            sig = _collect_signature(lines, i, "{")
            symbols.append({"kind": "pub fn", "name": _extract_name(after_vis[3:]), "signature": sig})
        elif after_vis.startswith("struct "):
            name = _extract_name(after_vis[7:])
            sig = f"pub struct {name}"
            # Collect generic bounds
            generic = _extract_generics(after_vis[7:])
            if generic:
                sig = f"pub struct {name}{generic}"
            fields = _collect_struct_fields(lines, i)
            if fields:
                sig += " { " + ", ".join(fields) + " }"
            symbols.append({"kind": "pub struct", "name": name, "signature": sig})
        elif after_vis.startswith("enum "):
            name = _extract_name(after_vis[5:])
            variants = _collect_enum_variants(lines, i)
            sig = f"pub enum {name}"
            if variants:
                sig += " { " + ", ".join(variants) + " }"
            symbols.append({"kind": "pub enum", "name": name, "signature": sig})
        elif after_vis.startswith("trait "):
            name = _extract_name(after_vis[6:])
            trait_methods = _collect_trait_methods(lines, i)
            sig = f"pub trait {name}"
            if trait_methods:
                sig += " { " + "; ".join(trait_methods) + "; }"
            symbols.append({"kind": "pub trait", "name": name, "signature": sig})
        elif after_vis.startswith("type "):
            symbols.append({"kind": "pub type", "name": _extract_name(after_vis[5:]), "signature": stripped})
        elif after_vis.startswith("const "):
            symbols.append({"kind": "pub const", "name": _extract_name(after_vis[6:]), "signature": stripped})

        brace_depth += line_opens
        i += 1

    return symbols


def _strip_visibility(line: str) -> str:
    """Strip pub/pub(crate)/pub(super) prefix."""
    if line.startswith("pub("):
        close = line.find(")")
        if close >= 0:
            return line[close + 1:].strip()
    if line.startswith("pub "):
        return line[4:].strip()
    return line


def _extract_generics(text: str) -> str:
    """Extract <...> generic parameters."""
    if "<" not in text:
        return ""
    depth = 0
    start = text.find("<")
    for i in range(start, len(text)):
        if text[i] == "<":
            depth += 1
        elif text[i] == ">":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return ""


def _extract_impl_name(line: str) -> str:
    """Extract type name from 'impl<T> Foo<T> for Bar'."""
    # Remove "impl" and optional generics
    rest = line[4:].strip()
    if rest.startswith("<"):
        close = rest.find(">")
        if close >= 0:
            rest = rest[close + 1:].strip()
    return _extract_name(rest)


def _extract_impl_methods(lines: List[str], start: int) -> List[Dict[str, str]]:
    """Extract pub fn declarations from inside an impl block."""
    methods = []
    depth = 0
    for i in range(start, min(start + 200, len(lines))):
        depth += lines[i].count("{") - lines[i].count("}")
        stripped = lines[i].strip()
        if (stripped.startswith("pub fn ") or stripped.startswith("pub(crate) fn ")):
            after_vis = _strip_visibility(stripped)
            sig = _collect_signature(lines, i, "{")
            methods.append({"kind": "pub fn", "name": _extract_name(after_vis[3:]), "signature": sig})
        if depth <= 0 and i > start:
            break
    return methods


def _collect_enum_variants(lines: List[str], start: int) -> List[str]:
    """Collect variant names from a Rust enum."""
    variants = []
    depth = 0
    for i in range(start, min(start + 50, len(lines))):
        depth += lines[i].count("{") - lines[i].count("}")
        stripped = lines[i].strip()
        if i > start and depth >= 1 and stripped and not stripped.startswith("//"):
            name = stripped.split("(")[0].split("{")[0].rstrip(",").strip()
            if name and name[0].isupper():
                variants.append(name)
        if depth <= 0 and i > start:
            break
    return variants[:15]


def _collect_trait_methods(lines: List[str], start: int) -> List[str]:
    """Collect fn signatures from a trait definition."""
    methods = []
    depth = 0
    for i in range(start, min(start + 50, len(lines))):
        depth += lines[i].count("{") - lines[i].count("}")
        stripped = lines[i].strip()
        if stripped.startswith("fn "):
            sig = stripped.rstrip(";").rstrip("{").strip()
            methods.append(sig)
        if depth <= 0 and i > start:
            break
    return methods[:10]


def _extract_python_symbols(lines: List[str]) -> List[Dict[str, str]]:
    """Extract top-level def and class from Python (non-underscore)."""
    symbols = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def ") and not stripped.startswith("def _"):
            name = _extract_name(stripped[4:])
            sig = stripped.rstrip(":")
            symbols.append({"kind": "def", "name": name, "signature": sig})
        elif stripped.startswith("class ") and not stripped.startswith("class _"):
            name = _extract_name(stripped[6:])
            symbols.append({"kind": "class", "name": name, "signature": stripped.rstrip(":")})
    return symbols


def _extract_generic_symbols(lines: List[str]) -> List[Dict[str, str]]:
    """Fallback: extract function/type declarations from any C-like language."""
    symbols = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("export ") or stripped.startswith("public "):
            symbols.append({"kind": "export", "name": stripped[:60], "signature": stripped[:80]})
        elif stripped.startswith("func "):
            symbols.append({"kind": "func", "name": _extract_name(stripped[5:]), "signature": stripped[:80]})
    return symbols


def _extract_name(text: str) -> str:
    """Extract identifier name from text (stops at non-alphanumeric)."""
    name = []
    for ch in text.strip():
        if ch.isalnum() or ch == "_":
            name.append(ch)
        else:
            break
    return "".join(name)


def _collect_signature(lines: List[str], start: int, terminator: str) -> str:
    """Collect a multi-line signature up to the terminator character."""
    sig_parts = []
    for i in range(start, min(start + 5, len(lines))):
        sig_parts.append(lines[i].strip())
        if terminator in lines[i]:
            break
    sig = " ".join(sig_parts)
    # Trim at the terminator
    idx = sig.find(terminator)
    if idx >= 0:
        sig = sig[:idx].strip()
    return sig


def _collect_struct_fields(lines: List[str], start: int) -> List[str]:
    """Collect pub field names from a Rust struct."""
    fields = []
    brace_depth = 0
    for i in range(start, min(start + 30, len(lines))):
        line = lines[i].strip()
        brace_depth += line.count("{") - line.count("}")
        if "pub " in line and ":" in line:
            field = line.strip().rstrip(",")
            fields.append(field)
        if brace_depth <= 0 and i > start:
            break
    return fields[:10]
