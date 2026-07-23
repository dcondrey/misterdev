"""Built-in, tool-free mutation of the CHANGED REGION — a suite-strength check.

``mutation_gate`` runs an EXTERNAL mutation tool (mutmut / Stryker / cargo-mutants)
the project must configure; most projects configure none, so it skips and offers
no protection. This module is the built-in complement: it mutates only the lines a
fix changed, using a handful of language-agnostic, syntax-preserving operators,
re-runs the project's OWN test command against each mutant, and reports how many
survive. A surviving mutant means the suite cannot tell the fix from a subtly
broken version — i.e. the passing test did not actually constrain the change.

Why the CHANGED region, not the whole file: it is cheap (a few test runs, not
thousands), and it answers the exact question a per-task gate cares about — "did
the test that just passed actually verify THIS fix?" That is the evasion-resistant
signal a shape heuristic (delete-and-collapse detection) cannot give: a padded or
gutted-body stub that "passes" leaves almost every changed-region mutant alive,
because the suite never pinned the behavior.

Deterministic operators (below) are chosen to preserve syntax, so a mutant that
fails the tests failed on BEHAVIOR, not a broken parse (which would inflate the
kill rate). Opt-in and best-effort: off unless ``orchestrator.changed_region_
mutation`` is set; ``min_score`` defaults to 0.0 (pure measurement — never RED)
and can be raised to gate. Equivalent mutants (a swap that cannot change any
observable behavior) inflate the survivor count, so treat a middling score as a
weak SIGNAL and reserve a hard floor for the high-confidence "zero killed" case.
"""

import difflib
import re
import subprocess
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple

from misterdev.core.execution.bounded import run_bounded
from misterdev.core.execution.outcomes import GREEN, RED, SKIP
from misterdev.core.verification.mutation_gate import MutationResult
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)


def _bump(m: "re.Match") -> str:
    return str(int(m.group(0)) + 1)


# (compiled operator, replacement). Replacement is a str (static swap) or a
# callable(match)->str. All are SYNTAX-PRESERVING in-place edits, so a mutant that
# breaks the tests broke behavior, not the parse. Bare `<`/`>` and arithmetic are
# deliberately excluded (they hit generics `List<T>`, arrows `=>`/`->`, and unary
# minus, which would break syntax and produce false kills).
_MUTATORS: Tuple[Tuple["re.Pattern", object], ...] = (
    (re.compile(r"(?<![<>=!])==(?!=)"), "!="),
    (re.compile(r"(?<![<>=!])!=(?!=)"), "=="),
    (re.compile(r"<="), ">="),
    (re.compile(r">="), "<="),
    (re.compile(r"&&"), "||"),
    (re.compile(r"\|\|"), "&&"),
    (re.compile(r"\btrue\b"), "false"),
    (re.compile(r"\bfalse\b"), "true"),
    (re.compile(r"\bTrue\b"), "False"),
    (re.compile(r"\bFalse\b"), "True"),
    (re.compile(r" and "), " or "),
    (re.compile(r" or "), " and "),
    (re.compile(r"\b\d+\b"), _bump),
)


def changed_line_indices(old_content: str, new_content: str) -> Set[int]:
    """0-based indices of lines in ``new_content`` that a diff added/changed vs
    ``old_content`` — the region a fix actually touched."""
    old = old_content.splitlines()
    new = new_content.splitlines()
    changed: Set[int] = set()
    for tag, _i1, _i2, j1, j2 in difflib.SequenceMatcher(
        None, old, new, autojunk=False
    ).get_opcodes():
        if tag in ("replace", "insert"):
            changed.update(range(j1, j2))
    if not changed and new and not old:
        # A brand-NEW file (no prior content): every line is new. An UNCHANGED file
        # (old == new) legitimately has no changed region — do not treat it as
        # fully changed, which would mutate untouched code.
        changed = set(range(len(new)))
    return changed


def generate_mutants(
    source: str, changed_lines: Set[int], max_mutants: int = 12
) -> List[Tuple[str, str]]:
    """Distinct single-operator mutants of ``source``, each applying ONE operator
    at ONE site on a changed line. Returns ``(mutated_source, description)`` pairs,
    capped at ``max_mutants``. Pure and deterministic."""
    lines = source.splitlines(keepends=True)
    mutants: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    for idx in sorted(changed_lines):
        if idx >= len(lines):
            continue
        line = lines[idx]
        for rx, repl in _MUTATORS:
            m = rx.search(line)
            if not m:
                continue
            piece = repl(m) if callable(repl) else repl
            new_line = line[: m.start()] + piece + line[m.end() :]
            if new_line == line:
                continue
            mutated = "".join(lines[:idx] + [new_line] + lines[idx + 1 :])
            if mutated == source or mutated in seen:
                continue
            seen.add(mutated)
            mutants.append(
                (mutated, f"L{idx + 1}: '{m.group(0).strip()}' -> '{piece.strip()}'")
            )
            if len(mutants) >= max_mutants:
                return mutants
    return mutants


def run_changed_region_mutation(
    project_root: Path,
    file_path: str,
    old_content: str,
    new_content: str,
    test_command: str,
    runner: Optional[Callable[[str, float], Tuple[bool, str]]] = None,
    min_score: float = 0.0,
    timeout: float = 300.0,
    max_mutants: int = 12,
) -> MutationResult:
    """Mutate the changed region of ``file_path`` (already holding ``new_content``,
    the fix that passed) and score how many mutants the test suite KILLS.

    ``runner(cmd, timeout) -> (ok, output)`` runs the suite; a mutant is KILLED when
    the suite fails on it (``ok`` False) and SURVIVES when it still passes. The file
    is restored to ``new_content`` afterward, always. GREEN when the kill rate meets
    ``min_score`` (default 0.0 -> always GREEN, pure measurement); RED below it.
    SKIP when there is nothing to run or no mutant to make.
    """
    if not test_command:
        return MutationResult(SKIP, reason="no test command")
    changed = changed_line_indices(old_content, new_content)
    mutants = generate_mutants(new_content, changed, max_mutants)
    if not mutants:
        return MutationResult(SKIP, reason="no mutants for the changed region")

    target = project_root / file_path
    run = runner or _make_local_runner(project_root)

    def _work() -> MutationResult:
        killed = 0
        try:
            for src, _desc in mutants:
                target.write_text(src, encoding="utf-8")
                ok, _out = run(test_command, timeout)
                if not ok:
                    killed += 1
        finally:
            # Restore the accepted fix no matter what — never leave a mutant on disk.
            target.write_text(new_content, encoding="utf-8")
        score = killed / len(mutants)
        status = GREEN if score >= min_score else RED
        reason = (
            f"{killed}/{len(mutants)} changed-region mutants killed "
            f"(score {score:.2f}); a low score means the suite does not verify the fix"
        )
        return MutationResult(status, score=score, reason=reason)

    return run_bounded(
        _work,
        timeout * len(mutants) + 10,
        MutationResult(SKIP, reason="timed out"),
        "Changed-region mutation",
    )


def _make_local_runner(
    project_root: Path,
) -> Callable[[str, float], Tuple[bool, str]]:
    def _run(cmd: str, timeout: float) -> Tuple[bool, str]:
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, ""
        return proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")

    return _run
