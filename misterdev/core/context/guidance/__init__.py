"""Per-language best-practice guidance injected into the edit system prompt.

Every language authors guidance as a list of :class:`Rule`s (see ``._rules``):
each edit injects only the rules relevant to the task — the ``core`` baseline
plus rules whose triggers match the task context (description, acceptance
criteria, error logs, code). Rule text is terse and symbolic to hold token cost
down; selection is the primary lever.

Extension resolution is deliberately independent of the executor's
``_detect_language``: that map folds ``.jsx``/``.tsx`` into javascript/typescript
and doesn't know ``.html``/``.css``, whereas guidance wants React for JSX/TSX and
dedicated HTML/CSS rule sets.
"""

from pathlib import Path
from typing import Iterable, Optional

from ._rules import render_rules, select_rules
from .cpp import CPP_RULES
from .csharp import CSHARP_RULES
from .css import CSS_RULES
from .elixir import ELIXIR_RULES
from .html import HTML_RULES
from .kotlin import KOTLIN_RULES
from .python import PYTHON_RULES
from .react import REACT_RULES
from .rust import RUST_RULES
from .swift import SWIFT_RULES
from .typescript import TYPESCRIPT_RULES

# Language name / alias (lowercased) -> (title, rule list).
_RULES = {
    "python": ("Python", PYTHON_RULES),
    "py": ("Python", PYTHON_RULES),
    "rust": ("Rust", RUST_RULES),
    "rs": ("Rust", RUST_RULES),
    "swift": ("Swift", SWIFT_RULES),
    "csharp": ("C#", CSHARP_RULES),
    "cs": ("C#", CSHARP_RULES),
    "c#": ("C#", CSHARP_RULES),
    "kotlin": ("Kotlin", KOTLIN_RULES),
    "kt": ("Kotlin", KOTLIN_RULES),
    "react": ("React", REACT_RULES),
    "html": ("HTML", HTML_RULES),
    "css": ("CSS", CSS_RULES),
    "typescript": ("TypeScript", TYPESCRIPT_RULES),
    "ts": ("TypeScript", TYPESCRIPT_RULES),
    "cpp": ("C++", CPP_RULES),
    "c++": ("C++", CPP_RULES),
    "cc": ("C++", CPP_RULES),
    "elixir": ("Elixir", ELIXIR_RULES),
    "ex": ("Elixir", ELIXIR_RULES),
}

# Source-file extension -> guidance key. More specific than a bare language name
# (a .tsx file wants React, not generic TypeScript).
_EXT_TO_KEY = {
    ".py": "python",
    ".pyi": "python",
    ".rs": "rust",
    ".swift": "swift",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".ts": "typescript",
    ".jsx": "react",
    ".tsx": "react",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".ex": "elixir",
    ".exs": "elixir",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "css",
    ".sass": "css",
}


def _resolve(key: str, context: str) -> str:
    """Selected-rule guidance for a resolved language key, or "" when unknown."""
    entry = _RULES.get(key)
    if entry is None:
        return ""
    title, rules = entry
    return render_rules(title, select_rules(rules, context))


def get_language_guidance(language: str, context: str = "") -> str:
    """Return guidance for a language name, or "" when none applies.

    ``context`` (task description, acceptance criteria, error logs, code) drives
    relevance selection.
    """
    if not language:
        return ""
    return _resolve(language.strip().lower(), context)


def guidance_for_files(
    target_files: Optional[Iterable[str]],
    fallback_language: str = "",
    context: str = "",
) -> str:
    """Pick the best-matching guidance for the files a task edits.

    Prefers a file-extension match (so ``.jsx`` picks React and ``.rs`` picks the
    Rust rules); falls back to the project language. Returns "" when nothing
    matches, so the caller can inject unconditionally.
    """
    for path in target_files or []:
        key = _EXT_TO_KEY.get(Path(path).suffix.lower())
        if key:
            resolved = _resolve(key, context)
            if resolved:
                return resolved
    return get_language_guidance(fallback_language, context)


__all__ = [
    "PYTHON_RULES",
    "RUST_RULES",
    "SWIFT_RULES",
    "CSHARP_RULES",
    "KOTLIN_RULES",
    "REACT_RULES",
    "HTML_RULES",
    "CSS_RULES",
    "TYPESCRIPT_RULES",
    "CPP_RULES",
    "ELIXIR_RULES",
    "get_language_guidance",
    "guidance_for_files",
]
