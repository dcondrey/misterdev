"""Deterministic, structure-aware compression of build/test error output.

Compiler and test output is machine-structured and mostly presentation noise —
source-line echoes, caret underlines, ``--explain`` hints, blank ``|`` gutters,
warning summaries. A capable model needs only ``location · code · message``.
Compressing before the output ever reaches the prompt collapses the verbose-rustc
explosion (a measured 446K tokens accumulated across nine retries on one trivial
task) to a handful of lines, losslessly for the actionable signal. Pure local
logic — no LLM, no network.
"""

import re
from typing import List, Optional

# The first errors usually cascade into the rest, and rustc even reports
# "aborting due to N previous errors"; keeping the first few is enough to fix.
_MAX_ITEMS = 8

# rustc: `error[E0308]: mismatched types` / `error: cannot find value ...`.
_RUSTC_HEAD = re.compile(r"^(error|warning)(\[[A-Z]\d+\])?:\s*(.+)$")
_RUSTC_LOC = re.compile(r"^\s*-->\s*(\S+)")

# Lines that carry no signal on their own: the source echo, caret underlines,
# blank gutters, `= note/help`, and the "for more information" explain hint.
_GENERIC_NOISE = re.compile(
    r"^\s*(\d+\s*\|| *\||\^+|~+|=\s|-->|\.{3}|"
    r"for more information about this error|"
    r"note: `#\[|note: this error originates|"
    r"warning: unused|Compiling |Finished |Running )",
    re.IGNORECASE,
)

# Generic error-signal patterns (pytest, tsc, clang, dotnet, go, javac, ...).
_ERROR_SIGNAL = re.compile(
    r"(?i)\b(error|failed|failure|panic|exception|assert|expected|"
    r"undefined|cannot find|mismatched|traceback|E\d{3,4}\b)"
)


def compress_error_log(
    output: str, language: Optional[str] = None, max_items: int = _MAX_ITEMS
) -> str:
    """Return a compact, deduped canonical form of compiler/test error output.

    Returns "" for empty input. Never raises — on anything unexpected it degrades
    to a bounded head of the raw output, so it can only ever shrink a prompt.
    """
    if not output or not output.strip():
        return ""
    lang = (language or "").lower()
    try:
        if "rust" in lang or "error[E" in output or "-->" in output:
            compact = _compress_rustc(output, max_items)
        else:
            compact = _compress_generic(output, max_items)
    except Exception:
        compact = []
    if not compact:
        compact = _fallback(output, max_items)
    return "\n".join(compact)


def _compress_rustc(output: str, max_items: int) -> List[str]:
    """Pair each `error[..]: msg` with its `--> file:line` and drop the rest."""
    lines = output.splitlines()
    out: List[str] = []
    seen = set()
    i = 0
    while i < len(lines) and len(out) < max_items:
        m = _RUSTC_HEAD.match(lines[i].strip())
        if not m or m.group(1) != "error":  # keep errors, skip warnings/notes
            i += 1
            continue
        code = (m.group(2) or "").strip("[]")
        msg = m.group(3).strip()
        if msg.startswith("aborting due to"):  # rustc summary line, not an error
            i += 1
            continue
        loc = ""
        for j in range(i + 1, min(i + 4, len(lines))):
            lm = _RUSTC_LOC.match(lines[j])
            if lm:
                loc = lm.group(1)
                break
        key = (loc, code, msg)
        if key not in seen:
            seen.add(key)
            head = f"{loc}: " if loc else ""
            out.append(f"{head}error{f'[{code}]' if code else ''}: {msg}")
        i += 1
    dropped = _count_rustc_errors(output) - len(out)
    if dropped > 0:
        out.append(f"... and {dropped} more error(s)")
    return out


def _compress_generic(output: str, max_items: int) -> List[str]:
    """Keep error-signal lines, drop source echo / ASCII art, dedup."""
    out: List[str] = []
    seen = set()
    for raw in output.splitlines():
        line = raw.rstrip()
        if not line.strip() or _GENERIC_NOISE.match(line):
            continue
        if not _ERROR_SIGNAL.search(line):
            continue
        norm = line.strip()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
        if len(out) >= max_items:
            out.append("... (further error lines omitted)")
            break
    return out


def _fallback(output: str, max_items: int) -> List[str]:
    """Bounded head of non-blank lines — worst case still shrinks the prompt."""
    lines = [ln.rstrip() for ln in output.splitlines() if ln.strip()]
    head = lines[: max_items * 2]
    if len(lines) > len(head):
        head.append(f"... ({len(lines) - len(head)} more lines)")
    return head


def _count_rustc_errors(output: str) -> int:
    m = re.search(r"aborting due to (\d+) previous error", output)
    if m:
        return int(m.group(1))
    return sum(
        1
        for ln in output.splitlines()
        if (h := _RUSTC_HEAD.match(ln.strip())) and h.group(1) == "error"
    )
