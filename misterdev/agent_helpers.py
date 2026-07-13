import time
from typing import Optional

from rich.console import Console

from misterdev.analyzers.project_analyzer import has_test_files
from misterdev.config import get_setting
from misterdev.core.verification.validator import gate_ran_no_tests
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)
console = Console()


class _WorktreeProjectView:
    """A Project facade that overrides only `path` (for worktree execution).

    Everything else (config, llm_client, topography, tools, env) delegates to
    the base project, so the executor reads shared context but writes, builds,
    and commits inside the worktree.
    """

    def __init__(self, base, path):
        self._base = base
        self.path = path

    def __getattr__(self, name):
        return getattr(self._base, name)


class ProgressReporter:
    """Lightweight wave/task progress logger for long runs."""

    def __init__(self, total_tasks: int):
        self.total = total_tasks
        self.completed = 0
        self.failed = 0
        self.current_wave = 0
        self.start_time = time.time()
        self._task_start: Optional[float] = None

    def start_wave(self, wave_num: int, task_ids: list[str]):
        self.current_wave = wave_num
        logger.info(f"=== Wave {wave_num} === [{', '.join(task_ids)}]")

    def start_task(self, task_id: str, title: str):
        self._task_start = time.time()
        logger.info(
            f"[{self.completed + self.failed}/{self.total}] Starting {task_id}: {title}"
        )

    def end_task(self, task_id: str, success: bool):
        elapsed = time.time() - self._task_start if self._task_start else 0
        if success:
            self.completed += 1
            logger.info(
                f"[{self.completed + self.failed}/{self.total}] {task_id} DONE ({elapsed:.0f}s)"
            )
        else:
            self.failed += 1
            logger.warning(
                f"[{self.completed + self.failed}/{self.total}] {task_id} FAILED ({elapsed:.0f}s)"
            )

    def summary(self):
        total_time = time.time() - self.start_time
        logger.info(
            f"=== Complete: {self.completed} done, {self.failed} failed, {total_time:.0f}s total ==="
        )


def _combine_commands(*cmds: Optional[str]) -> Optional[str]:
    """Join shell commands with ``&&`` (each parenthesised), or None if all empty.

    Used to run the visible suite and the golden suite as one pass/fail check
    so the existing single-command gate/bisect logic enforces both.
    """
    present = [c for c in cmds if c]
    if not present:
        return None
    return " && ".join(f"({c})" for c in present)


def _budget_exhausted(client) -> bool:
    """True when the client reports a non-positive remaining budget.

    Defensive: a non-numeric ``budget_remaining`` (e.g. a test double) is not
    treated as exhausted.
    """
    remaining = getattr(client, "budget_remaining", None)
    return (
        isinstance(remaining, (int, float))
        and not isinstance(remaining, bool)
        and remaining <= 0
    )


def _apply_budget_ceiling(client, flag_budget: float) -> None:
    """Set the client budget to the tighter of its config budget and the flag.

    The project.yaml budget is already on the client; both it and the --budget
    flag are ceilings, so the minimum wins and a config cap is never silently
    overridden by the CLI default. Falls back to the flag when the current
    budget isn't numeric (e.g. a test double).
    """
    current = getattr(client, "_budget", None)
    if isinstance(current, (int, float)) and not isinstance(current, bool):
        client._budget = min(current, flag_budget)
    else:
        client._budget = flag_budget


def _warn_if_baseline_broken(assessment, report) -> None:
    """Loudly surface a failing baseline build before any work begins.

    A red baseline silently disables the integration gate (it needs a green
    baseline to detect regressions), so the run would execute largely ungated.
    Make that visible and record it rather than letting it pass quietly.
    """
    health = assessment.health
    if health.builds:
        return
    head = (health.build_output or "").strip()[:600]
    msg = (
        "Baseline build FAILS — the integration gate disables itself without a "
        "green baseline, so this run will be largely ungated. Fix the build first "
        "(consider debug mode). Build error:\n" + (head or "(no output captured)")
    )
    logger.error(msg)
    console.print(f"[red]Baseline build is failing.[/] {head[:200]}")
    report.key_decisions.append(
        "WARNING: baseline build was failing at start; gates degraded for this run"
    )


def _warn_if_no_test_gate(assessment, project, report) -> None:
    """Loudly surface that a run will proceed with NO test gate while the project
    has a test suite — the safety hole that let an ungated run rewrite existing
    tests while reporting build OK. Detection covers most layouts; this catches
    the residual case where a suite exists but no command was resolved.
    """
    if assessment.structure.test_command:
        return
    # Multi-target repos gate per sub-project, so an empty top-level command is
    # not "no gate": declared targets with a build/test command (or auto-target
    # discovery) provide the protection. Don't cry wolf in that case.
    cfg = getattr(project, "config", {}) or {}
    targets = cfg.get("targets") or []
    # `targets` is an open, non-schema-validated config list, so a malformed
    # project.yaml could hold non-dict entries; guard so this advisory warning
    # never aborts the build with an AttributeError.
    if any(
        isinstance(t, dict) and (t.get("test_command") or t.get("build_command"))
        for t in targets
    ):
        return
    if get_setting(cfg, "orchestrator", "auto_targets"):
        return
    if not has_test_files(project.path):
        return
    msg = (
        "No test command was detected, but this project HAS test files. The run "
        "will proceed with only a build/syntax gate, so existing tests will NOT "
        "protect against regressions and edits to them will not be caught. Set "
        "`test_command` in project.yaml to enable the test gate."
    )
    logger.warning(msg)
    console.print(f"[yellow]No test gate:[/] {msg}")
    report.degraded_subsystems.append(
        "No test gate: test files exist but no test command was detected"
    )


def _warn_if_test_gate_is_noop(assessment, report) -> None:
    """Surface a resolved test command that runs but exercises ZERO tests.

    Distinct from ``_warn_if_no_test_gate`` (a MISSING command): here a command
    exists and exits clean yet collects nothing — a wrong path, a marker filter
    that matches no tests, or a misinferred runner. That is a false-GREEN gate: it
    passes every edit while catching no regression, the same silent hole a missing
    gate opens. Fires only on high confidence — an explicit "no tests ran" signal
    AND a parsed count of 0 — so a real suite whose output format we don't parse
    never trips it, and a healthy multi-crate run (cargo prints "running 0 tests"
    per empty crate) is excluded by the nonzero total.
    """
    health = assessment.health
    if not assessment.structure.test_command:
        return  # a missing command is _warn_if_no_test_gate's concern
    if not health.tests_pass:
        return  # a failing baseline is surfaced by _warn_if_baseline_broken
    if health.test_count > 0:
        return  # it ran real tests
    if not gate_ran_no_tests(health.test_output):
        return  # a 0 count alone may just be an unparsed format; need the signal
    msg = (
        "Test command resolved but ran ZERO tests — the test gate is a no-op and "
        "will pass every edit without catching regressions. Check the command "
        "targets the right path/markers (detected: "
        f"`{assessment.structure.test_command}`)."
    )
    logger.warning(msg)
    console.print(f"[yellow]No-op test gate:[/] {msg}")
    report.degraded_subsystems.append(
        "No-op test gate: test command resolved but ran zero tests"
    )


def _check_golden_config(config) -> None:
    """Warn when the golden suite is half-configured (a silent integrity hole).

    The two halves are independent: ``golden_paths`` protects+conceals the
    files, ``golden_command`` enforces them as a gate. Configuring one without
    the other silently drops a guarantee, so surface it loudly.
    """
    orch = config.get("orchestrator", {})
    paths = orch.get("golden_paths") or []
    command = orch.get("golden_command")
    if command and not paths:
        logger.warning(
            "golden_command is set but golden_paths is empty: golden tests are "
            "enforced as a gate but NOT protected from edits; the model could "
            "weaken them. Set golden_paths to the same files."
        )
    if paths and not command:
        logger.warning(
            "golden_paths is set but golden_command is empty: golden files are "
            "protected and hidden but never run as a gate. Set golden_command "
            "to enforce them."
        )
