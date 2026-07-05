"""Line-based heuristic public-symbol extraction for Rust source.

Fallback used when the tree-sitter Rust grammar is unavailable.
"""

from typing import Dict, List

from ._text import _extract_name


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
            symbols.append(
                {
                    "kind": "pub fn",
                    "name": _extract_name(after_vis[3:]),
                    "signature": sig,
                }
            )
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
            symbols.append(
                {
                    "kind": "pub type",
                    "name": _extract_name(after_vis[5:]),
                    "signature": stripped,
                }
            )
        elif after_vis.startswith("const "):
            symbols.append(
                {
                    "kind": "pub const",
                    "name": _extract_name(after_vis[6:]),
                    "signature": stripped,
                }
            )

        brace_depth += line_opens
        i += 1

    return symbols


def _strip_visibility(line: str) -> str:
    """Strip pub/pub(crate)/pub(super) prefix."""
    if line.startswith("pub("):
        close = line.find(")")
        if close >= 0:
            return line[close + 1 :].strip()
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
                return text[start : i + 1]
    return ""


def _extract_impl_name(line: str) -> str:
    """Extract type name from 'impl<T> Foo<T> for Bar'."""
    # Remove "impl" and optional generics
    rest = line[4:].strip()
    if rest.startswith("<"):
        close = rest.find(">")
        if close >= 0:
            rest = rest[close + 1 :].strip()
    return _extract_name(rest)


def _extract_impl_methods(lines: List[str], start: int) -> List[Dict[str, str]]:
    """Extract pub fn declarations from inside an impl block."""
    methods = []
    depth = 0
    for i in range(start, min(start + 200, len(lines))):
        depth += lines[i].count("{") - lines[i].count("}")
        stripped = lines[i].strip()
        if stripped.startswith("pub fn ") or stripped.startswith("pub(crate) fn "):
            after_vis = _strip_visibility(stripped)
            sig = _collect_signature(lines, i, "{")
            methods.append(
                {
                    "kind": "pub fn",
                    "name": _extract_name(after_vis[3:]),
                    "signature": sig,
                }
            )
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
