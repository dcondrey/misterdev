"""Preflight checks for an unattended run — the ``misterdev doctor`` command.

An autonomous run spends real budget and edits real code, so it is worth
confirming the project is actually ready BEFORE it starts: the tree is clean and
on the base branch, no leftover ``task/*`` branches or dangling worktrees, the
configured models resolve, a throwaway worktree primes and its toolchain
resolves, and any REQUIREMENTS.md inputs are answered.

Each check yields a ``Check(name, status, detail, fix)`` where status is
``pass`` / ``warn`` / ``fail``. Only a HARD blocker (something an unattended run
cannot recover from — a dirty tree, a stranded branch, an unroutable model) is a
``fail``; degradations that the run self-heals or falls back on are ``warn``.
``aggregate`` turns the checks into counts and the process exit code (non-zero
iff any hard failure). The check builders take already-gathered inputs, so the
routing and the aggregation are pure and directly unit-testable; the orchestrator
gathers the inputs and the CLI renders the checklist.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

PASS = "pass"
WARN = "warn"
FAIL = "fail"


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str = ""
    fix: str = ""


def aggregate(checks: List[Check]) -> Dict:
    """Counts + process exit code for a checklist. Exit is non-zero iff any check
    is a hard ``fail`` — warnings never block an unattended run, they inform it."""
    passed = sum(1 for c in checks if c.status == PASS)
    warnings = sum(1 for c in checks if c.status == WARN)
    failures = sum(1 for c in checks if c.status == FAIL)
    return {
        "passed": passed,
        "warnings": warnings,
        "failures": failures,
        "ready": failures == 0,
        "exit_code": 1 if failures else 0,
    }


def check_git_repo(is_git: bool) -> Check:
    if is_git:
        return Check("git repository", PASS, "project is a git repo")
    return Check(
        "git repository",
        WARN,
        "not a git repo — parallel worktree isolation is unavailable",
        "run `git init` (and commit a baseline) for isolated parallel execution",
    )


def check_clean_tree(dirty: str) -> Check:
    if not dirty:
        return Check("clean working tree", PASS, "no uncommitted changes")
    return Check(
        "clean working tree",
        FAIL,
        f"uncommitted changes present: {dirty[:120]}",
        "commit or stash your changes (a run carries/reverts branch-per-task work)",
    )


def check_on_base_branch(current: Optional[str], base: str) -> Check:
    if not current or current in ("HEAD", "(detached)"):
        return Check(
            "on base branch",
            FAIL,
            "HEAD is detached",
            f"`git checkout {base}` so tasks branch from and merge into the base",
        )
    if current.startswith("task/") or current.startswith("doctor/"):
        return Check(
            "on base branch",
            FAIL,
            f"HEAD is on a leftover run branch '{current}'",
            f"`git checkout {base}` and delete the stranded branch",
        )
    if current != base:
        return Check(
            "on base branch",
            WARN,
            f"HEAD is on '{current}', not the base '{base}'",
            f"`git checkout {base}` if you meant to run against the base branch",
        )
    return Check("on base branch", PASS, f"on '{base}'")


def check_leftover_task_branches(task_branches: List[str]) -> Check:
    if not task_branches:
        return Check("no leftover task branches", PASS, "none found")
    shown = ", ".join(task_branches[:5]) + (" …" if len(task_branches) > 5 else "")
    return Check(
        "no leftover task branches",
        WARN,
        f"{len(task_branches)} leftover task/* branch(es): {shown}",
        "delete them (`git branch -D <name>`); a run uses unique names so they are safe to drop",
    )


def check_dangling_worktrees(worktrees: List[str]) -> Check:
    if not worktrees:
        return Check("no dangling worktrees", PASS, "none found")
    return Check(
        "no dangling worktrees",
        WARN,
        f"{len(worktrees)} orchestrator worktree(s) still registered",
        "run `git worktree prune` (a run prunes on start, but cleaning now avoids confusion)",
    )


def check_models(ok: bool, detail: str) -> Check:
    if ok:
        return Check("models resolve", PASS, f"model preflight ok: {detail}")
    return Check(
        "models resolve",
        FAIL,
        f"model preflight failed: {detail}",
        "set a valid model id in project.yaml (build.model) that your provider serves",
    )


def check_worktree_prime(
    primed_ok: Optional[bool], healthcheck_ok: Optional[bool], detail: str = ""
) -> Check:
    """``primed_ok``/``healthcheck_ok`` are True/False, or None when that step did
    not apply (no setup/health command for this project)."""
    if primed_ok is None and healthcheck_ok is None:
        return Check(
            "worktree prime + healthcheck",
            PASS,
            "no dependency prime needed for this project",
        )
    if healthcheck_ok is False:
        return Check(
            "worktree prime + healthcheck",
            WARN,
            f"toolchain did not resolve in a throwaway worktree: {detail[:120]}",
            "check the dependency install / toolchain (a run re-primes once, then flags it)",
        )
    if primed_ok is False:
        return Check(
            "worktree prime + healthcheck",
            WARN,
            f"dependency prime failed: {detail[:120]}",
            "verify the install command; a run falls back to the gate's own install",
        )
    return Check(
        "worktree prime + healthcheck",
        PASS,
        "a throwaway worktree primed and its toolchain resolved",
    )


def check_evolution_configured(benchmark_dir: Optional[str]) -> Check:
    """Informational only: evolution is opt-in and unrelated to build readiness,
    so both states are ``pass`` — this never blocks or warns."""
    if benchmark_dir:
        return Check(
            "evolution configured",
            PASS,
            f"benchmark_dir: {benchmark_dir}",
        )
    return Check(
        "evolution configured",
        PASS,
        "not configured (opt-in; set `evolution.benchmark_dir` in project.yaml "
        "or pass --benchmark to `misterdev evolve`)",
    )


def check_requirements(unsatisfied: List[str]) -> Check:
    if not unsatisfied:
        return Check("requirements satisfied", PASS, "no outstanding inputs")
    shown = ", ".join(unsatisfied[:6]) + (" …" if len(unsatisfied) > 6 else "")
    return Check(
        "requirements satisfied",
        WARN,
        f"{len(unsatisfied)} REQUIREMENTS.md input(s) not yet provided: {shown}",
        "answer them in .orchestrator/REQUIREMENTS.md (or set the env vars), or run with --proceed to park them",
    )
