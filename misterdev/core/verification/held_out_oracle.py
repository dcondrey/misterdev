"""Held-out oracle: is a per-task acceptance check actually NON-TRIVIAL?

The strongest per-task oracle is a test that fails before the fix and passes
after (see spec_tests.py). But a generated or hand-written acceptance test can be
*trivial* — it passes even against a do-nothing implementation (e.g. an assert
that the function merely returns a string). A trivial oracle is how a
runtime-broken fix still "passes" and merges (docs/research-directions.md,
Theme 1 — Cogeneration 2601.19066).

This module gives the missing guarantee: STUB the fix's changed region (blank the
edited function bodies) and re-run the acceptance test. If the test still passes
against the stub, it does not actually constrain the behavior — reject it as a
weak oracle rather than trust the pass. If the stub fails the test, the oracle is
real: passing it means something.

Pure + injectable (a ``runner`` runs the test; a ``writer`` swaps file contents),
so it is fully testable with no subprocess. Best-effort and self-restoring: the
original file is always put back, and any failure degrades to "cannot judge"
(SKIP) rather than raising into the build.
"""

import ast
import re
from pathlib import Path
from typing import Callable, Optional, Tuple

from misterdev.core.execution.outcomes import GREEN, RED, SKIP
from misterdev.core.verification.changed_region_mutation import changed_line_indices
from misterdev.core.verification.mutation_gate import MutationResult
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)

# Runner: (test_command, timeout) -> (passed, output). Same shape as the
# changed-region mutation runner, so callers can share one.
Runner = Callable[[str, float], Tuple[bool, str]]


def stub_python(source: str, changed_lines) -> Optional[str]:
    """Blank the body of every function/method that overlaps a changed line,
    replacing it with ``raise NotImplementedError``. Returns None when nothing
    could be stubbed (no changed function) or the source does not parse."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    lines = source.splitlines(keepends=True)
    changed = set(changed_lines)
    # Collect (start, end, indent) for each edited function, outermost-first so a
    # later replacement never lands inside an already-replaced span.
    spans = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start = node.body[0].lineno - 1  # 0-based first body line
        end = (node.end_lineno or node.body[-1].lineno) - 1
        if any(start <= ln <= end for ln in changed):
            indent = " " * (node.body[0].col_offset)
            spans.append((start, end, indent))
    if not spans:
        return None
    spans.sort(key=lambda s: s[0], reverse=True)  # replace bottom-up
    for start, end, indent in spans:
        lines[start : end + 1] = [f"{indent}raise NotImplementedError\n"]
    return "".join(lines)


def _stub_for(path: str, old: str, new: str) -> Optional[str]:
    changed = changed_line_indices(old, new)
    if path.endswith(".py"):
        return stub_python(new, changed)
    return None  # only Python stubbing is implemented; other languages SKIP


def check_oracle(
    project_path: Path,
    rel_path: str,
    old_content: str,
    new_content: str,
    test_command: str,
    runner: Runner,
    timeout: float = 120.0,
) -> MutationResult:
    """RED when the acceptance test still PASSES against a stubbed fix (trivial
    oracle — the test does not constrain behavior); GREEN when the stub fails it
    (real oracle). SKIP when no stub can be formed or no test command exists.
    Always restores the file."""
    if not test_command:
        return MutationResult(SKIP, reason="no test command")
    stub = _stub_for(rel_path, old_content, new_content)
    if stub is None or stub == new_content:
        return MutationResult(SKIP, reason="no stubbable changed function")
    target = project_path / rel_path
    try:
        target.write_text(stub, encoding="utf-8")
        passed, _out = runner(test_command, timeout)
    except OSError as e:
        return MutationResult(SKIP, reason=f"oracle check error: {e}")
    finally:
        try:
            target.write_text(new_content, encoding="utf-8")
        except OSError as e:  # restoration must not be silent — it changed source
            logger.error(f"held-out oracle: FAILED to restore {rel_path}: {e}")
    if passed:
        return MutationResult(
            RED,
            score=0.0,
            reason="acceptance test passes against a do-nothing stub — a trivial "
            "oracle that does not verify the fix",
        )
    return MutationResult(
        GREEN, score=1.0, reason="oracle rejects the stub (non-trivial)"
    )


_TRIVIAL_ASSERT_RE = re.compile(
    r"assert\s+(isinstance\(|.*\bis\s+not\s+None\b|len\(.*\)\s*>?=?\s*\d|True\b)",
)


def looks_trivial(test_source: str) -> bool:
    """Cheap static smell test: a test whose only assertions are shape checks
    (isinstance / not-None / len>0 / assert True) is likely a weak oracle. A
    zero-cost pre-filter before the (more expensive) stub run."""
    asserts = [ln for ln in test_source.splitlines() if "assert" in ln]
    if not asserts:
        return True
    return all(_TRIVIAL_ASSERT_RE.search(ln) for ln in asserts)
