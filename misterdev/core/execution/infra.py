"""Classify a gate failure as an ENVIRONMENT/infrastructure fault, not a code bug.

Some gate failures say nothing about the code under test: a command timed out, a
dependency was never installed, the package store was locked by a concurrent
install, the disk filled, the box ran out of memory/file handles. Reflecting on
these as if they were code defects wastes a model call and can revert correct
work; the right response is to repair the environment and re-run, not to edit.

This is the counterpart to ``blocker.py``: ``blocker`` detects an EXTERNAL
resource the user must supply (park the task); ``infra`` detects a TRANSIENT
local fault the orchestrator should self-heal by retrying. Both are pure,
signal-only, and conservative — a plain assertion or type error must never be
misread as infrastructure.
"""

import re
from typing import Optional, Tuple

# (compiled signal, human reason). Every phrase is one an ENVIRONMENT fault emits,
# never something a normal type/assertion failure prints, so a real code error is
# not misclassified as infra (at worst a false positive costs one extra re-run).
_SIGNALS: Tuple[Tuple[re.Pattern, str], ...] = (
    (
        re.compile(r"timed out after \d+s|\bETIMEDOUT\b|operation timed out", re.I),
        "a command timed out",
    ),
    (
        re.compile(
            r"cannot find module|\bMODULE_NOT_FOUND\b|ERR_MODULE_NOT_FOUND|"
            r"is not the tsc command|\bcommand not found\b|"
            r"no such file or directory, open .*node_modules",
            re.I,
        ),
        "a dependency was not installed in the worktree",
    ),
    (
        re.compile(
            r"\bELOCK\b|store is locked|waiting for the lock|"
            r"another process is running|lock file .* held",
            re.I,
        ),
        "the package store was locked by a concurrent install",
    ),
    (
        re.compile(r"\bENOSPC\b|no space left on device", re.I),
        "the disk is full",
    ),
    (
        re.compile(r"\bEMFILE\b|\bENFILE\b|too many open files", re.I),
        "the process ran out of file handles",
    ),
    (
        re.compile(
            r"out of memory|JavaScript heap out of memory|\bSIGKILL\b|"
            r"\bENOMEM\b|Killed\s*$",
            re.I,
        ),
        "the process ran out of memory or was killed",
    ),
)


def infra_failure(output: str) -> Optional[str]:
    """A short human reason when ``output`` shows an environment/infra fault, else
    None. None means "treat as an ordinary (code) failure"."""
    text = output or ""
    for pattern, reason in _SIGNALS:
        if pattern.search(text):
            return reason
    return None
