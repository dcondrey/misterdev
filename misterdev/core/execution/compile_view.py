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
from typing import Callable, Dict, List, Optional


@dataclass
class CompileError:
    """One compiler diagnostic, reduced to what the model needs to fix it."""

    message: str
    code: Optional[str] = None  # e.g. "E0308"
    location: Optional[str] = None  # "file:line:col"
    detail: Optional[str] = None  # e.g. "expected `u32`, found `&str`"


@dataclass(frozen=True)
class CompilerAdapter:
    """A per-language compiler-diagnostic adapter.

    ``language`` is the canonical lowercase language key ("rust", "typescript").
    ``name`` is the tool the parser targets ("rustc", "tsc"). ``parse`` turns raw
    compiler output into ``CompileError`` records (empty when it recognizes
    nothing). ``detect`` reports whether this adapter's compiler produced the
    output, so a caller with no language hint can route by content. New languages
    register an adapter instead of extending an if/else on language name.
    """

    language: str
    name: str
    parse: Callable[[str], List[CompileError]]
    detect: Callable[[str], bool]


_REGISTRY: Dict[str, CompilerAdapter] = {}


def register_adapter(adapter: CompilerAdapter) -> None:
    """Register (or replace) the adapter for ``adapter.language``."""
    _REGISTRY[adapter.language] = adapter


def get_adapter(language: Optional[str]) -> Optional[CompilerAdapter]:
    """The adapter for a language key, or None when none is registered."""
    return _REGISTRY.get((language or "").lower())


def registered_languages() -> List[str]:
    """The languages that currently have a registered adapter."""
    return sorted(_REGISTRY)


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


def _detect_rustc(output: str) -> bool:
    return bool(re.search(r"^error(?:\[E\d+\])?: ", output, re.M)) and "-->" in output


# --- tsc (TypeScript) -------------------------------------------------------

# Classic tsc: `src/app.ts(12,5): error TS2322: Type 'string' is not ...`
_TSC_CLASSIC = re.compile(
    r"^(?P<file>.+?)\((?P<line>\d+),(?P<col>\d+)\):\s*"
    r"error\s+(?P<code>TS\d+):\s*(?P<message>.+)$"
)
# Pretty tsc (`--pretty`): `src/app.ts:12:5 - error TS2322: Type 'string' ...`
_TSC_PRETTY = re.compile(
    r"^(?P<file>\S+):(?P<line>\d+):(?P<col>\d+)\s*-\s*"
    r"error\s+(?P<code>TS\d+):\s*(?P<message>.+)$"
)


def _parse_tsc(output: str) -> List[CompileError]:
    out: List[CompileError] = []
    for line in output.splitlines():
        m = _TSC_CLASSIC.match(line) or _TSC_PRETTY.match(line)
        if not m:
            continue
        out.append(
            CompileError(
                message=m.group("message").strip(),
                code=m.group("code"),
                location=f"{m.group('file').strip()}:{m.group('line')}:{m.group('col')}",
            )
        )
    return out


def _detect_tsc(output: str) -> bool:
    return bool(re.search(r"error\s+TS\d+:", output))


# --- go (go build / go vet) -------------------------------------------------

# `./main.go:10:6: undefined: helper` — the column is optional.
_GO_ERROR = re.compile(
    r"^(?P<file>\S+\.go):(?P<line>\d+):(?:(?P<col>\d+):)?\s+(?P<message>.+)$"
)


def _parse_go(output: str) -> List[CompileError]:
    out: List[CompileError] = []
    for line in output.splitlines():
        m = _GO_ERROR.match(line)
        if not m:
            continue
        loc = f"{m.group('file')}:{m.group('line')}"
        if m.group("col"):
            loc += f":{m.group('col')}"
        out.append(CompileError(message=m.group("message").strip(), location=loc))
    return out


def _detect_go(output: str) -> bool:
    return bool(re.search(r"^\S+\.go:\d+:", output, re.M))


# --- swiftc -----------------------------------------------------------------

# `/src/App.swift:12:15: error: cannot find 'foo' in scope`
_SWIFT_ERROR = re.compile(
    r"^(?P<file>.+?\.swift):(?P<line>\d+):(?P<col>\d+):\s*error:\s*(?P<message>.+)$"
)


def _parse_swift(output: str) -> List[CompileError]:
    out: List[CompileError] = []
    for line in output.splitlines():
        m = _SWIFT_ERROR.match(line)
        if not m:
            continue
        out.append(
            CompileError(
                message=m.group("message").strip(),
                location=f"{m.group('file')}:{m.group('line')}:{m.group('col')}",
            )
        )
    return out


def _detect_swift(output: str) -> bool:
    return bool(re.search(r"\.swift:\d+:\d+:\s*error:", output))


# --- csc / dotnet (MSBuild) -------------------------------------------------

# `Program.cs(12,20): error CS0103: <message>` with an optional trailing
# ` [project.csproj]` MSBuild tag, which is stripped from the message.
_CSC_ERROR = re.compile(
    r"^(?P<file>.+?)\((?P<line>\d+),(?P<col>\d+)\):\s*error\s+(?P<code>CS\d+):\s*"
    r"(?P<message>.+?)(?:\s*\[[^\]]+\])?$"
)


def _parse_csharp(output: str) -> List[CompileError]:
    out: List[CompileError] = []
    for line in output.splitlines():
        m = _CSC_ERROR.match(line)
        if not m:
            continue
        out.append(
            CompileError(
                message=m.group("message").strip(),
                code=m.group("code"),
                location=f"{m.group('file')}:{m.group('line')}:{m.group('col')}",
            )
        )
    return out


def _detect_csharp(output: str) -> bool:
    return bool(re.search(r"error\s+CS\d+:", output))


register_adapter(CompilerAdapter("rust", "rustc", _parse_rustc, _detect_rustc))
register_adapter(CompilerAdapter("typescript", "tsc", _parse_tsc, _detect_tsc))
register_adapter(CompilerAdapter("go", "go build", _parse_go, _detect_go))
register_adapter(CompilerAdapter("swift", "swiftc", _parse_swift, _detect_swift))
register_adapter(CompilerAdapter("csharp", "csc", _parse_csharp, _detect_csharp))


def extract_compile_errors(
    output: str, language: Optional[str] = None
) -> List[CompileError]:
    """Parse compiler output into diagnostic records. Empty on unrecognized output.

    Routes by the explicit ``language`` when its adapter recognizes the output;
    otherwise falls back to content detection across the registry, so a caller
    with the wrong or no language hint still gets the right diagnostics.
    """
    if not output:
        return []
    adapter = get_adapter(language)
    if adapter is not None:
        errors = adapter.parse(output)
        if errors:
            return errors
    for candidate in _REGISTRY.values():
        if candidate.detect(output):
            errors = candidate.parse(output)
            if errors:
                return errors
    return []


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
