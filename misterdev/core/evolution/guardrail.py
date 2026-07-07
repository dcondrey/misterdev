"""The reward-hacking wall — the non-negotiable guardrail.

The #1 failure mode of a self-improving loop is not weak capability, it is the
agent editing its own judge: the benchmark evaluator, the gate suite, or the
held-out tests. A diff there can raise the measured score without improving
anything real (Goodhart / reward hacking). These paths are walled off exactly
like golden paths: the mutation operator MUST call :func:`assert_mutation_allowed`
before applying any candidate diff, and a candidate that touches them is refused
outright — it never reaches the fitness function.

Also refuses any path that escapes the repository (``..`` traversal or an
absolute path), so a proposed edit can never reach outside the sandbox.
"""

from typing import Iterable, List

# Prefixes (repo-relative, POSIX) that a self-edit may never target. Walling off
# the evaluator + gates + held-out tests closes the reward-hacking hole; walling
# off the evolution package itself stops the loop rewriting its own fitness rule,
# archive, or this guardrail.
_PROTECTED_PREFIXES = (
    "evaluation/",  # the benchmark harness — the judge itself
    "misterdev/core/verification/",  # the gate suite (correctness/mutation/spec gates)
    "misterdev/core/evolution/",  # the loop's own fitness / archive / guardrail
    "tests/",  # held-out + regression tests
)


class ProtectedPathError(Exception):
    """Raised when a candidate mutation targets a walled-off or escaping path."""


def _normalize(path: str) -> str:
    """Repo-relative POSIX form of ``path`` for prefix matching.

    Strips a leading ``./``, converts backslashes, and collapses redundant
    separators. Does NOT resolve against the filesystem (the repo may not be the
    cwd); traversal is detected structurally in :func:`_escapes_repo`.
    """
    p = str(path).strip().replace("\\", "/").lstrip("./")
    while "//" in p:
        p = p.replace("//", "/")
    return p


def _escapes_repo(path: str) -> bool:
    """True if ``path`` is absolute or walks out of the repo via ``..``.

    Any such target is refused: a self-edit must stay inside the sandboxed
    worktree, never reach a sibling repo, the home dir, or system files.
    """
    raw = str(path).strip().replace("\\", "/")
    if raw.startswith("/") or (len(raw) > 1 and raw[1] == ":"):  # POSIX / Windows abs
        return True
    return any(seg == ".." for seg in raw.split("/"))


def is_protected(path: str) -> bool:
    """True if ``path`` may not be mutated (walled off, or escapes the repo)."""
    if _escapes_repo(path):
        return True
    norm = _normalize(path)
    return any(
        norm == prefix.rstrip("/") or norm.startswith(prefix)
        for prefix in _PROTECTED_PREFIXES
    )


def assert_mutation_allowed(paths: Iterable[str]) -> None:
    """Raise :class:`ProtectedPathError` if any of ``paths`` is off-limits.

    The single chokepoint the mutation operator calls before applying a diff, so
    a candidate can never touch the evaluator, the gates, the held-out tests, or
    anything outside the repo — the difference between targeted improvement and
    reward hacking.
    """
    bad: List[str] = sorted({str(p) for p in paths if is_protected(p)})
    if bad:
        raise ProtectedPathError(
            f"mutation targets walled-off or escaping paths: {bad}"
        )
