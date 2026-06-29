import time
from typing import Optional

from rich.console import Console

from my_project_orchestrator.analyzers.project_analyzer import has_test_files
from my_project_orchestrator.config import get_setting
from my_project_orchestrator.logging_setup import setup_logger

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
    return isinstance(remaining, (int, float)) and remaining <= 0


def _apply_budget_ceiling(client, flag_budget: float) -> None:
    """Set the client budget to the tighter of its config budget and the flag.

    The project.yaml budget is already on the client; both it and the --budget
    flag are ceilings, so the minimum wins and a config cap is never silently
    overridden by the CLI default. Falls back to the flag when the current
    budget isn't numeric (e.g. a test double).
    """
    current = getattr(client, "_budget", None)
    if isinstance(current, (int, float)):
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
    if any(t.get("test_command") or t.get("build_command") for t in targets):
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
