"""Structured extraction of the exact compiler diagnostics from a failed build.

The test-failure counterpart, :mod:`.failure_view`, hands the model the exact
assertion (expected vs actual) so it fixes what it can precisely see. Compile
errors had no equivalent: the resolver extracted a location (``--> file:line``)
but dropped the *diagnostic* — the error code, the message, and the ``expected X,
found Y`` span. On compiled languages (rust especially) the model then iterates
on a lossy view of the compiler, which is the measured cost driver on hard rust
exercises. This seam parses a compiler's output into ``CompileError`` records
(code, message, location, the expected/found detail) and renders a tight block
that leads the retry context.

Deterministic and offline-testable: parsers are validated against captured real
compiler output. Unknown output yields nothing (the caller keeps its existing
compressed view), so this can only add signal, never remove the fallback.
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CompileError:
    """One compiler diagnostic, reduced to what the model needs to fix it."""

    message: str
    code: Optional[str] = None  # e.g. "E0308"
    location: Optional[str] = None  # "file:line:col"
    detail: Optional[str] = None  # e.g. "expected `u32`, found `&str`"


# --- rustc ------------------------------------------------------------------

# `error[E0308]: mismatched types` / `error: cannot find value ...` (no code).
_RUSTC_ERROR = re.compile(r"^error(?:\[(?P<code>E\d+)\])?: (?P<message>.+)$")
# ` --> src/lib.rs:2:18`
_RUSTC_LOC = re.compile(r"^\s*-->\s*(?P<loc>\S+:\d+:\d+)\s*$")
# The caret-annotation carrying the useful detail: `^^^ expected `u32`, found
# `&str`` or `^^^ not found in this scope`. The leading `---`/`|` framing has no
# caret, so requiring `^` isolates the informative line.
_RUSTC_DETAIL = re.compile(r"\^+\s*(?P<detail>\S.*\S)\s*$")


def _parse_rustc(output: str) -> List[CompileError]:
    lines = output.splitlines()
    out: List[CompileError] = []
    for i, line in enumerate(lines):
        m = _RUSTC_ERROR.match(line)
        if not m:
            continue
        e = CompileError(message=m.group("message").strip(), code=m.group("code"))
        # Scan this diagnostic's block (until the next `error`/blank run) for its
        # location and the first informative caret line.
        for w in lines[i + 1 : i + 12]:
            if _RUSTC_ERROR.match(w):
                break
            loc = _RUSTC_LOC.match(w)
            if loc and e.location is None:
                e.location = loc.group("loc")
                continue
            d = _RUSTC_DETAIL.search(w)
            if (
                d
                and e.detail is None
                and ("expected" in d.group("detail") or "found" in d.group("detail"))
            ):
                e.detail = d.group("detail")
        out.append(e)
    return out


_COMPILERS = {"rustc": _parse_rustc}


def _detect_compiler(output: str) -> Optional[str]:
    if re.search(r"^error(?:\[E\d+\])?: ", output, re.M) and "-->" in output:
        return "rustc"
    return None


def extract_compile_errors(
    output: str, language: Optional[str] = None
) -> List[CompileError]:
    """Parse compiler output into diagnostic records. Empty on unrecognized output."""
    if not output:
        return []
    compiler = {"rust": "rustc"}.get((language or "").lower())
    if (
        compiler is None
        or compiler not in _COMPILERS
        or not _COMPILERS[compiler](output)
    ):
        compiler = _detect_compiler(output)
    if compiler is None:
        return []
    return _COMPILERS[compiler](output)


def render_compile_view(errors: List[CompileError], max_errors: int = 5) -> str:
    """Render the exact compiler diagnostics as a tight, lead-with-truth block.

    Empty when there is nothing structured to show, so the caller keeps its
    existing compressed error context unchanged.
    """
    if not errors:
        return ""
    shown = errors[:max_errors]
    lines = [f"{len(errors)} compile error(s); the exact diagnostics:"]
    for e in shown:
        code = f"[{e.code}] " if e.code else ""
        loc = f" ({e.location})" if e.location else ""
        lines.append(f"- {code}{e.message}{loc}")
        if e.detail:
            lines.append(f"    {e.detail}")
    if len(errors) > max_errors:
        lines.append(f"  (+{len(errors) - max_errors} more)")
    return "\n".join(lines)
