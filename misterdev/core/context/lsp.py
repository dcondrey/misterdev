"""Optional LSP-based semantic diagnostics.

A language server surfaces errors the tree-sitter syntax gate cannot see —
undefined names, type mismatches, unresolved imports — across languages without
per-project linter configuration. This is strictly best-effort: the work runs in
a daemon worker thread with a hard timeout and returns ``None`` on any problem
(multilspy missing, unsupported language, server slow/unavailable), so it can
never block or slow a build. It is off unless ``orchestrator.lsp_diagnostics``
is enabled, and even then a timeout silently skips the check rather than failing.
"""

from pathlib import Path
from typing import Dict, List, Optional

from misterdev.core.execution.bounded import run_bounded
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)

# Our internal language ids -> multilspy code_language values. Languages without
# a multilspy server (c, cpp, swift) are intentionally absent -> gate skipped.
_LANG_MAP: Dict[str, str] = {
    "python": "python",
    "rust": "rust",
    "typescript": "typescript",
    "javascript": "javascript",
    "csharp": "csharp",
    "kotlin": "kotlin",
    "java": "java",
    "go": "go",
    "ruby": "ruby",
}

_LSP_SEVERITY_ERROR = 1  # LSP DiagnosticSeverity.Error

# Per-file diagnostic settle wait bounds (seconds). The actual wait scales down
# with file count so the total stays under the caller's hard timeout.
_MIN_FILE_WAIT = 0.3
_MAX_FILE_WAIT = 1.5


def _per_file_wait(num_files: int, settle_budget: float) -> float:
    """Settle wait per file: the budget split across files, clamped to bounds.

    Keeps the single-file wait close to the original 1.5s but shrinks it as the
    file set grows so ``wait * num_files`` cannot overrun the gate timeout (which
    previously caused every diagnostic to be skipped on larger projects).
    """
    if num_files <= 0:
        return _MIN_FILE_WAIT
    return max(_MIN_FILE_WAIT, min(_MAX_FILE_WAIT, settle_budget / num_files))


_LANG_EXTS: Dict[str, tuple] = {
    "python": (".py",),
    "rust": (".rs",),
    "typescript": (".ts", ".tsx"),
    "javascript": (".js", ".jsx", ".mjs", ".cjs"),
    "csharp": (".cs",),
    "kotlin": (".kt", ".kts"),
    "java": (".java",),
    "go": (".go",),
    "ruby": (".rb",),
}

_SKIP_DIRS = frozenset(
    {".venv", "venv", ".git", "node_modules", "__pycache__", "target", "build", "dist"}
)


def find_source_files(project_root: Path, language: str, cap: int = 40) -> List[str]:
    """Project-relative source paths for ``language`` (bounded by ``cap``)."""
    exts = _LANG_EXTS.get((language or "").lower())
    if not exts:
        return []
    root = Path(project_root)
    out: List[str] = []
    for path in sorted(root.rglob("*")):
        if (
            path.suffix in exts
            and path.is_file()
            and not (_SKIP_DIRS & set(path.parts))
        ):
            out.append(str(path.relative_to(root)))
            if len(out) >= cap:
                break
    return out


def collect_diagnostics(
    project_root: Path,
    language: str,
    rel_files: List[str],
    timeout: float = 30.0,
) -> Optional[List[dict]]:
    """Return error-severity diagnostics for ``rel_files``, or None if skipped.

    None means "no opinion" (unsupported language, multilspy unavailable, or the
    server did not respond within ``timeout``); callers must treat it as a
    no-op, never a pass/fail signal. A list (possibly empty) means the server
    ran: each item is ``{"file", "line", "message"}`` for an error diagnostic.
    """
    # Swift has no multilspy server (see _LANG_MAP), so route it to the direct
    # sourcekit-lsp adapter, which returns the same {file,line,message} shape.
    if (language or "").lower() == "swift" and rel_files:
        from misterdev.core.context.lsp_swift import diagnostics_for

        by_file = diagnostics_for(str(project_root), rel_files, timeout)
        out = [
            {"file": rel, "line": d["line"], "message": d["message"]}
            for rel, diags in by_file.items()
            for d in (diags or [])
        ]
        return out or None

    code_lang = _LANG_MAP.get((language or "").lower())
    if code_lang is None or not rel_files:
        return None

    # Bound the in-server settle time to a fraction of the hard timeout so the
    # per-file waits can't sum past it and force a blanket SKIP (the old fixed
    # 1.5s * N did exactly that for ~20+ files at the 30s default).
    settle_budget = max(timeout * 0.7, _MIN_FILE_WAIT)

    def _work() -> Optional[List[dict]]:
        try:
            return _collect(project_root, code_lang, rel_files, settle_budget)
        except Exception as e:  # multilspy/server failures are non-fatal
            logger.debug(f"LSP diagnostics unavailable ({language}): {e}")
            return None

    # A hung server is abandoned to its daemon thread and the gate skips (None),
    # so the build is never blocked.
    return run_bounded(_work, timeout, None, f"LSP diagnostics ({language})")


def _collect(
    project_root: Path,
    code_lang: str,
    rel_files: List[str],
    settle_budget: float = 60.0,
) -> List[dict]:
    import asyncio

    from multilspy import LanguageServer
    from multilspy.multilspy_config import MultilspyConfig
    from multilspy.multilspy_logger import MultilspyLogger

    per_file = _per_file_wait(len(rel_files), settle_budget)

    async def _main() -> List[dict]:
        config = MultilspyConfig.from_dict({"code_language": code_lang})
        server = LanguageServer.create(config, MultilspyLogger(), str(project_root))
        captured: List[dict] = []
        async with server.start_server():
            # multilspy discards publishDiagnostics by default; override the
            # handler to capture them (diagnostics are server-pushed, not a
            # request/response).
            server.server.on_notification(
                "textDocument/publishDiagnostics", captured.append
            )
            for rel in rel_files:
                try:
                    with server.open_file(rel):
                        await asyncio.sleep(per_file)
                except Exception:
                    continue
        return _to_errors(captured)

    return asyncio.run(_main())


def _to_errors(captured: List[dict]) -> List[dict]:
    errors: List[dict] = []
    for params in captured:
        uri = params.get("uri", "")
        for diag in params.get("diagnostics", []):
            if diag.get("severity") == _LSP_SEVERITY_ERROR:
                line = diag.get("range", {}).get("start", {}).get("line", 0) + 1
                errors.append(
                    {
                        "file": uri.replace("file://", ""),
                        "line": line,
                        "message": diag.get("message", ""),
                    }
                )
    return errors


def format_lsp_context(diagnostics: Optional[List[dict]], cap: int = 20) -> str:
    """Render collected LSP diagnostics as a prompt-injectable block.

    Turns the semantic-error list from :func:`collect_diagnostics` into context
    the editor can read while fixing code — richer than raw compiler stderr
    because it is per-file, per-line, and semantic (unresolved symbols, type
    mismatches). Returns "" for None (LSP had no opinion) or an empty list, so
    the caller can inject unconditionally. Bounded to ``cap`` lines so a flood
    of diagnostics can't dominate the prompt.
    """
    if not diagnostics:
        return ""
    lines = ["## Language-server diagnostics (semantic errors to resolve):"]
    for d in diagnostics[:cap]:
        loc = f"{d.get('file', '?')}:{d.get('line', '?')}"
        lines.append(f"- {loc}: {d.get('message', '').strip()}")
    remaining = len(diagnostics) - cap
    if remaining > 0:
        lines.append(f"- ... and {remaining} more")
    return "\n".join(lines)


def collect_and_format_lsp_context(
    project_root, language: str, rel_files: List[str], timeout: float = 30.0
) -> str:
    """Collect semantic diagnostics for the edited files and render them.

    Convenience over :func:`collect_diagnostics` + :func:`format_lsp_context` for
    the editor's retry context. Returns "" when the LSP has no opinion
    (unsupported language, server absent, or timed out), so a caller can append
    it unconditionally. Timeout-bounded via ``collect_diagnostics``.
    """
    diagnostics = collect_diagnostics(Path(project_root), language, rel_files, timeout)
    return format_lsp_context(diagnostics)
