"""Module-level helpers and constants for the markdown plan executor.

Moved verbatim out of the original ``markdown_plan_executor.py`` so the
executor mixins and the package ``__init__`` can share them. Pure code
movement: behaviour is identical.
"""

import ast
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

from misterdev.logging_setup import setup_logger
from misterdev.utils.file_utils import is_golden_path

# Historical private alias re-exported from this module's namespace; the
# executor and tests import ``_is_golden_path`` from here.
_is_golden_path = is_golden_path

# ``__package__`` of this submodule is the package path that the original
# single-file module had as ``__name__``, so the logger name is unchanged.
logger = setup_logger(__package__)
# Appended to every code-generation prompt. The model edits large files by
# emitting only the changed regions as anchored SEARCH/REPLACE hunks instead of
# rewriting the whole file (which truncates past the output-token limit). The
# parser (responses.parse_search_replace_blocks) recognizes exactly this shape;
# whole-file blocks remain valid for short new files with no markers.
EDIT_FORMAT_INSTRUCTIONS = """

## Output format (required)
Edit existing files with anchored SEARCH/REPLACE blocks. Do NOT reprint the
whole file. For each change, put the file path on the fence line, then one or
more blocks:

```<lang>:<path>
<<<<<<< SEARCH
<exact lines copied verbatim from the current file, whitespace included>
=======
<the replacement lines>
>>>>>>> REPLACE
```

Rules:
- The SEARCH text must match the current file exactly and identify exactly one
  location; include enough surrounding lines to make it unique.
- COPY the SEARCH lines verbatim from the file shown in Code Context — do not
  retype from memory, reformat, or guess; a single wrong character fails the edit.
- Use several small blocks rather than one large one.
- To ADD code to an existing file (a new function, import, or export), anchor on
  a real adjacent line you can see in the file — e.g. an existing function near
  where the new code belongs — and put that anchor verbatim in BOTH SEARCH and
  REPLACE, with your new code added in REPLACE. Never anchor on a line you have
  not seen in the file.
- To create a NEW file, use a single block with an empty SEARCH section and the
  full file contents in the REPLACE section.
"""

# Used after anchored SEARCH/REPLACE edits repeatedly fail to APPLY (the model's
# SEARCH text doesn't match the file). A full-file rewrite needs no anchoring, so
# it always applies — converting a no-progress stall into an applied edit the
# real gates can then give feedback on.
FULL_FILE_FALLBACK_INSTRUCTIONS = """

## Output format (required)
Your anchored SEARCH/REPLACE edits failed to apply because the SEARCH text did
not match the file. STOP using SEARCH/REPLACE. Instead output the COMPLETE,
updated content of each file you change, as a single fenced code block per file
with the file path on the fence line:

```<lang>:<path>
<the entire file, from the first line to the last, with your changes integrated>
```

Reproduce the whole file verbatim except for your intended change. Do not omit,
summarize, or elide any existing code.
"""

# Maps file extensions to language identifiers for syntax validation and
# contract extraction. Unknown extensions fall back to "text".
_LANG_MAP = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hh": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".sh": "shell",
    ".bash": "shell",
}


def _relevant_line_ranges(symbols, task, n_lines: int):
    """Line ranges to show in full for a large target file.

    Picks symbols whose name appears in the task text, pads each by a few
    lines, and always includes the file head (imports/uses). Returns merged
    0-based inclusive ranges, or None when nothing matched so the caller can
    fall back to sending the whole file.
    """
    head_lines = 30
    margin = 3
    text = ""
    if task is not None:
        text = f"{task.description or ''} {task.acceptance_criteria or ''}"
    # Whole-token match (case-insensitive): a symbol is relevant only when its
    # name appears as a word in the task, not as a substring — otherwise short
    # names like "Loc" match inside unrelated words and pull in the whole file.
    tokens = {t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)}
    relevant = [
        s for s in symbols if s.name and len(s.name) >= 3 and s.name.lower() in tokens
    ]
    if not relevant:
        return None
    ranges = [
        (max(0, s.start_line - margin), min(n_lines - 1, s.end_line + margin))
        for s in relevant
    ]
    ranges.append((0, min(head_lines - 1, n_lines - 1)))
    return _merge_ranges(ranges)


def _merge_ranges(ranges):
    """Merge overlapping/adjacent (start, end) inclusive ranges, sorted."""
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _window_lines(lines: List[str], ranges) -> str:
    """Render only ``ranges`` of ``lines`` verbatim, with elision markers between.

    Kept spans are emitted unchanged so SEARCH/REPLACE can anchor against them;
    gaps become a marker that points back to the outline.
    """
    out = []
    prev_end = -1
    for start, end in ranges:
        if start > prev_end + 1:
            gap = start - (prev_end + 1)
            out.append(
                f"... [{gap} lines elided: L{prev_end + 2}-L{start} — see outline] ..."
            )
        out.extend(lines[start : end + 1])
        prev_end = end
    if prev_end < len(lines) - 1:
        gap = len(lines) - 1 - prev_end
        out.append(
            f"... [{gap} lines elided: L{prev_end + 2}-L{len(lines)} — see outline] ..."
        )
    return "\n".join(out)


def _bisect_first_failing(n: int, passes_at) -> int:
    """Binary-search [0, n) for the first index where passes_at(i) is False.

    Assumes a monotonic pass->fail boundary (all-pass prefix, then failures).
    Returns n-1 if nothing fails; callers should re-check that index.
    """
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if passes_at(mid):
            lo = mid + 1
        else:
            hi = mid
    return lo


# finish_reason values that mean the model hit its output-token limit and the
# response is incomplete. Anthropic reports "max_tokens"; OpenAI-compatible
# providers report "length". Compared case-insensitively.
_TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens"})


def _is_truncated(finish_reason: Optional[str]) -> bool:
    """True when ``finish_reason`` indicates the model ran out of output tokens."""
    return bool(finish_reason) and finish_reason.lower() in _TRUNCATED_FINISH_REASONS


def _detect_language(file_path: str) -> str:
    """Detect a source language from a file path's extension.

    Returns "text" for extensions with no known language mapping so callers
    fall back to language-agnostic validation rather than guessing.
    """
    ext = Path(file_path).suffix.lower()
    return _LANG_MAP.get(ext, "text")


# Path patterns that mark a file as holding tests, covering the languages in
# _LANG_MAP. Used to decide which edits get the tamper-resistance check.
_TEST_FILE_PATTERNS = (
    re.compile(r"(^|/)test_[^/]+\.py$"),
    re.compile(r"_test\.py$"),
    re.compile(r"(^|/)conftest\.py$"),
    re.compile(r"\.test\.(js|jsx|ts|tsx)$"),
    re.compile(r"\.spec\.(js|jsx|ts|tsx)$"),
    re.compile(r"_test\.go$"),
    re.compile(r"_test\.(rs|rb)$"),
    re.compile(r"(^|/)test_[^/]+\.(rb|c|cpp|cc)$"),
    re.compile(r"Test[^/]*\.java$"),
    # Swift XCTest, Kotlin/JUnit, C# (xUnit/NUnit/MSTest) test files.
    re.compile(r"Tests?\.(swift|kt|cs)$"),
    re.compile(r"(^|/)[Tt]ests?/"),
)

# Skip/ignore markers per language. Their COUNT must not grow across an edit:
# more skips is the cheapest way to make a suite "pass" by not running it.
_SKIP_PATTERNS = (
    re.compile(
        r"@(?:pytest\.mark\.skip|pytest\.mark\.skipif|unittest\.skip"
        r"|unittest\.SkipTest|skip|skipif)\b"
    ),
    re.compile(r"\b(?:it|describe|test|context)\.skip\b"),
    re.compile(r"\bxit\b|\bxdescribe\b"),
    re.compile(r"#\[ignore\b"),
    re.compile(r"\bt\.Skip(?:Now)?\b|\bt\.SkipNow\b"),
    re.compile(r"@(?:Ignore|Disabled)\b"),
    re.compile(r"\.skip\s*\(|\.todo\s*\("),
    # Swift XCTest skip; C# xUnit/NUnit/MSTest skip+ignore.
    re.compile(r"\bXCTSkip\b|\btry\s+XCTSkip"),
    re.compile(r"\[Ignore\b|\bSkip\s*=\s*[\"']"),
)

# Things that count as a "test" definition per language. We compare the total
# across all patterns rather than per-pattern so an edit can't dodge the check
# by converting one form of test to another.
_TEST_DEF_PATTERNS = (
    re.compile(r"(?m)^\s*def\s+test\w*\s*\("),
    re.compile(r"\b(?:it|test)\s*\(\s*['\"`]"),
    re.compile(r"#\[test\]"),
    re.compile(r"(?m)^\s*func\s+Test\w*\s*\("),
    re.compile(r"@Test\b"),
    # Swift XCTest methods (func testFoo) and C# attribute-based tests.
    re.compile(r"(?m)^\s*func\s+test\w*\s*\("),
    re.compile(r"\[(?:Fact|Theory|Test|TestMethod)\b"),
)

# Assertion-ish forms. Weakening a test often keeps the function but guts its
# checks, so a drop in assertion count is just as suspicious as a dropped test.
_ASSERT_PATTERNS = (
    re.compile(r"(?m)^\s*assert\b"),
    re.compile(r"\bself\.assert\w+\s*\("),
    re.compile(r"\bexpect\s*\("),
    re.compile(r"\bassert(?:_eq|_ne|_matches)?!\s*\("),
    re.compile(r"\bpytest\.raises\b|\bassertRaises\b"),
)

# Trivially-true assertions: a cheap way to keep the assertion COUNT steady
# while removing the actual check ("assert True", "expect(true).toBe(true)").
# An increase in these across an edit is treated as weakening, cross-language.
_TAUTOLOGY_PATTERNS = (
    re.compile(r"(?m)^\s*assert\s+(?:True|1)\s*(?:,|$)"),
    re.compile(r"\bself\.assertTrue\s*\(\s*True\s*\)"),
    re.compile(r"\bself\.assertFalse\s*\(\s*False\s*\)"),
    re.compile(r"\bassert!\s*\(\s*true\s*\)"),
    re.compile(r"\bexpect\s*\(\s*true\s*\)\s*\.\s*to(?:Be|Equal)\s*\(\s*true\s*\)"),
    re.compile(r"\bassert\s+1\s*===?\s*1\b"),
)

# In-body skip calls (vs the decorator forms in _SKIP_PATTERNS): inserting one
# of these short-circuits a test at runtime while leaving it visibly defined.
_INBODY_SKIP_PATTERNS = (
    re.compile(r"\bpytest\.skip\s*\("),
    re.compile(r"\bself\.skipTest\s*\("),
    re.compile(r"\bunittest\.SkipTest\b"),
    re.compile(r"\bt\.Skip(?:Now)?\s*\("),
)

# Fraction of the build budget that must remain before the LLM acceptance judge
# (default-on) is allowed to spend; below it, free-text criteria pass for free.
JUDGE_MIN_BUDGET_FRACTION = 0.1


def _is_test_file(file_path: str) -> bool:
    """True if the path names a test file in one of the supported languages."""
    norm = file_path.replace("\\", "/")
    return any(p.search(norm) for p in _TEST_FILE_PATTERNS)


# Known test/build runners that mark the START of a runnable acceptance command.
# We only treat acceptance_criteria as a command when one of these verbs appears,
# so free-text criteria ("the login form rejects empty passwords") are never
# mis-parsed into a command. Ordered/anchored so multi-word runners match before
# their first word (e.g. "python -m pytest" before bare "python").
_ACCEPTANCE_RUNNERS = (
    # Allow an absolute/relative interpreter path (e.g. a venv's python) before
    # "-m pytest", so a venv-pinned acceptance command is recognized too.
    r"(?:[\w./-]*/)?python3?\s+-m\s+pytest",
    r"pytest",
    r"cargo\s+test",
    r"cargo\s+build",
    r"cargo\s+check",
    r"cargo\s+clippy",
    r"go\s+test",
    r"go\s+build",
    r"npm\s+test",
    r"npm\s+run\s+\S+",
    r"npx\s+\S+",
    r"yarn\s+\S+",
    r"pnpm\s+\S+",
    r"make\s+\S+",
    r"make",
    r"ruff(?:\s+\S+)?",
    r"mypy",
    r"pyright",
    r"tsc",
    r"jest",
    r"vitest",
    r"tox",
    r"phpunit",
    r"rspec",
    r"gradle\s+\S+",
    r"\./gradlew\s+\S+",
    r"mvn\s+\S+",
)

# A runnable command starts at a known runner and runs to the end of the line (or
# a sentence-terminating boundary). Anything before the runner (e.g. "Verify that
# ") is dropped. The trailing tail is trimmed by _extract_acceptance_command.
_ACCEPTANCE_COMMAND_RE = re.compile(
    r"(?P<cmd>(?:" + "|".join(_ACCEPTANCE_RUNNERS) + r")[^\n]*)",
    re.IGNORECASE,
)

# Trailing prose that commonly follows a quoted command and is not part of it,
# e.g. "pytest tests/test_auth.py passes". Stripped from the extracted command.
_ACCEPTANCE_TAIL_RE = re.compile(
    r"\s+(?:passes?|succeeds?|should\s+pass|must\s+pass|exits?\s+0|"
    r"returns?\s+0|is\s+green|all\s+green|cleanly|without\s+errors?)\b.*$",
    re.IGNORECASE,
)


def _extract_acceptance_command(criteria: str) -> Optional[str]:
    """Extract a single runnable command from an acceptance-criteria string.

    Conservative: returns a command only when the text begins (after optional
    lead-in prose) with a known test/build runner. Trailing prose like
    "... passes" is trimmed so the runner sees just the command. Returns None
    for free-text criteria so un-parseable sentences never fail a task.
    """
    if not criteria:
        return None
    # Prefer a command fenced in backticks if present, but still require it to
    # start with a known runner so prose in backticks isn't run blindly.
    for candidate in re.findall(r"`([^`]+)`", criteria):
        m = _ACCEPTANCE_COMMAND_RE.match(candidate.strip())
        if m:
            return _ACCEPTANCE_TAIL_RE.sub("", m.group("cmd")).strip()
    m = _ACCEPTANCE_COMMAND_RE.search(criteria)
    if not m:
        return None
    cmd = m.group("cmd").strip()
    # Cut at the first sentence boundary so a trailing English sentence on the
    # same line doesn't get fed to the shell.
    cmd = re.split(r"(?<=\S)[.;]\s+[A-Z]", cmd, maxsplit=1)[0].strip()
    cmd = _ACCEPTANCE_TAIL_RE.sub("", cmd).strip()
    return cmd or None


def _test_metrics(content: str) -> Tuple[int, int, int]:
    """Cheap structural metrics for a test file: (tests, asserts, skips).

    Deterministic regex counts only, no parsing. Used to compare a test file
    before vs after an edit; growth is fine, shrinkage/more-skips is tamper.
    """
    tests = sum(len(p.findall(content)) for p in _TEST_DEF_PATTERNS)
    asserts = sum(len(p.findall(content)) for p in _ASSERT_PATTERNS)
    skips = sum(
        len(p.findall(content)) for p in (*_SKIP_PATTERNS, *_INBODY_SKIP_PATTERNS)
    )
    return tests, asserts, skips


def _count_tautologies(content: str) -> int:
    """Count trivially-true assertions (regex, cross-language)."""
    return sum(len(p.findall(content)) for p in _TAUTOLOGY_PATTERNS)


def _diagnose_tampering(before: str, after: str) -> Optional[str]:
    """Return a reason string if `after` weakens the test file, else None.

    Tampering = fewer tests, fewer assertions, more skip markers (decorator or
    in-body), or more trivially-true assertions. Pure additions (new
    tests/assertions) are allowed and return None.
    """
    bt, ba, bs = _test_metrics(before)
    at, aa, as_ = _test_metrics(after)
    reasons = []
    if at < bt:
        reasons.append(f"test count dropped {bt}->{at}")
    if aa < ba:
        reasons.append(f"assertion count dropped {ba}->{aa}")
    if as_ > bs:
        reasons.append(f"skip/ignore markers increased {bs}->{as_}")
    btaut, ataut = _count_tautologies(before), _count_tautologies(after)
    if ataut > btaut:
        reasons.append(f"trivially-true assertions increased {btaut}->{ataut}")
    return "; ".join(reasons) if reasons else None


def _ast_call_name(func: ast.AST) -> Optional[str]:
    """Best-effort dotted-call leaf name: ``self.assertEqual`` -> assertEqual."""
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _assertion_key(node: ast.AST) -> Optional[str]:
    """A structure-based identity for one assertion, or None if it isn't one.

    Keys on what the assertion checks (the AST of the condition / call args),
    not on its position or enclosing function, so the same check has the same
    key wherever it lives.
    """
    if isinstance(node, ast.Assert):
        return "assert:" + ast.dump(node.test)
    if isinstance(node, ast.Call):
        name = _ast_call_name(node.func)
        if name and (name.startswith("assert") or name in ("expect", "raises")):
            args = ",".join(ast.dump(a) for a in node.args)
            return f"{name}:{args}"
    return None


def _py_assertion_multiset(content: str) -> Optional["Counter"]:
    """Structural multiset of every assertion inside Python ``test*`` functions.

    Because assertions are keyed by structure (see ``_assertion_key``) rather
    than by their enclosing test's name or order, renaming, moving, reordering,
    splitting, or merging tests leaves the multiset unchanged. Only an assertion
    that disappears without an identical one reappearing is a real loss of
    coverage. Returns None when the content does not parse.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None
    counts: Counter = Counter()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test"):
            continue
        for sub in ast.walk(node):
            key = _assertion_key(sub)
            if key:
                counts[key] += 1
    return counts


def _parametrize_count(content: str) -> int:
    """Count ``parametrize``-style decorators (a legit way to reshape asserts)."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return 0
    total = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if _ast_call_name(target) == "parametrize":
                total += 1
    return total


def _py_skip_count(content: str) -> int:
    return sum(
        len(p.findall(content)) for p in (*_SKIP_PATTERNS, *_INBODY_SKIP_PATTERNS)
    )


def _diagnose_py_tampering(before: str, after: str) -> Optional[str]:
    """Python tamper diagnosis keyed on assertion survival, not test identity.

    Flags only genuine weakening: assertions that vanish without an identical
    check reappearing anywhere, an increase in trivially-true assertions, or
    more skip markers. Renames, moves, reorders, splits, and merges all
    preserve the assertion multiset and pass. Parametrization legitimately
    reshapes assertions, so a net loss is not held against an edit that adds a
    parametrize decorator. Returns None when either revision fails to parse.
    """
    before_asserts = _py_assertion_multiset(before)
    after_asserts = _py_assertion_multiset(after)
    if before_asserts is None or after_asserts is None:
        return None
    reasons = []
    if _parametrize_count(after) <= _parametrize_count(before):
        lost = sum((before_asserts - after_asserts).values())
        if lost:
            reasons.append(f"{lost} assertion(s) removed or weakened")
    if _count_tautologies(after) > _count_tautologies(before):
        reasons.append("trivially-true assertions increased")
    if _py_skip_count(after) > _py_skip_count(before):
        reasons.append("skip markers increased")
    return "; ".join(reasons) if reasons else None
