"""Zero-argument interactive menu — the "I don't remember the flags" entry.

``misterdev`` with no arguments (or ``misterdev interactive`` / ``misterdev i``)
launches a guided session: pick an action from a numbered menu, answer a couple
of prompts with sensible defaults, confirm, and it runs. No command sequence to
memorize and no LLM call required (unlike the natural-language route). Known
subcommands and the NL route still work unchanged for power users.

Dependency-free: plain ``input()`` + rich formatting. Ctrl-C / Ctrl-D exit
cleanly at any prompt.
"""

from pathlib import Path
from typing import List, Optional

from rich.console import Console

console = Console()


class _Quit(Exception):
    """Raised by a prompt on EOF/Ctrl-C to unwind to a clean exit."""


def _prompt(text: str, default: str = "") -> str:
    suffix = f" [dim]({default})[/]" if default else ""
    try:
        console.print(f"[bold cyan]?[/] {text}{suffix}", end=" ")
        reply = input().strip()
    except (EOFError, KeyboardInterrupt):
        raise _Quit from None
    return reply or default


def _confirm(text: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    reply = _prompt(f"{text} [dim]({hint})[/]").lower()
    if not reply:
        return default
    return reply in ("y", "yes")


def _menu(title: str, options: List[str]) -> int:
    console.print(f"\n[bold]{title}[/]")
    for i, opt in enumerate(options, 1):
        console.print(f"  [cyan]{i}[/]. {opt}")
    while True:
        reply = _prompt("Choose", default="1")
        if reply.isdigit() and 1 <= int(reply) <= len(options):
            return int(reply) - 1
        console.print("[red]  Pick a number from the list.[/]")


def _ask_project() -> str:
    path = _prompt("Project path", default=".")
    p = Path(path).expanduser()
    if not p.exists():
        console.print(
            f"[yellow]  {p} does not exist yet (a new project will be created).[/]"
        )
    return str(p)


def _do_build(orchestrator, mode_goal: Optional[str]) -> None:
    project = _ask_project()
    goal = mode_goal or _prompt("What should misterdev do? (describe the goal)")
    if not goal:
        console.print("[yellow]  No goal given; skipping.[/]")
        return
    budget = _prompt("Budget in dollars", default="100")
    args = f"{goal} --budget {budget}"
    label = mode_goal or goal
    if not _confirm(f"Run build on [cyan]{project}[/] — “{label}” at ${budget}?"):
        return
    console.print("[green]  Building…[/]")
    result = orchestrator.build(project, args)
    if result:
        console.print(result)


def _do_run(orchestrator) -> None:
    project = _ask_project()
    tasks = _prompt("Task-list file (blank = use the devplan/ directory)")
    tasklist = tasks or None
    # Always preview first — the safe default for an unfamiliar list.
    console.print("[green]  Planning…[/]")
    orchestrator.run_project(project, dry_run=True, tasklist=tasklist)
    if _confirm("Execute this plan now?", default=True):
        orchestrator.run_project(project, tasklist=tasklist)


def _do_status(orchestrator) -> None:
    project = _ask_project()
    status = orchestrator.get_project_status(project)
    if "error" in status:
        console.print(f"[red]  Error:[/] {status['error']}")
        return
    console.print(f"\n[bold]{status['name']}[/]  [dim]{status['path']}[/]")
    tasks = status.get("tasks") or []
    if not tasks:
        console.print("[dim]  No tasks found.[/]")
        return
    for t in tasks:
        console.print(f"  [{t.get('status', '?')}] {t.get('id', '?')}")


def _do_report(orchestrator) -> None:
    from misterdev.cli import _print_report

    _print_report(_ask_project())


def run_interactive(orchestrator) -> int:
    """Run the guided menu loop; returns a process exit code."""
    console.print(
        "\n[bold]misterdev[/] — autonomous build orchestrator\n"
        "[dim]Pick an action; press Ctrl-C to quit.[/]"
    )
    actions = [
        (
            "Build — describe a goal; misterdev writes/fixes/completes the code",
            lambda: _do_build(orchestrator, None),
        ),
        (
            "Run a task list — execute a plan file (any format) or the devplan/ dir",
            lambda: _do_run(orchestrator),
        ),
        (
            "Debug — find and fix everything that's broken",
            lambda: _do_build(orchestrator, "debug"),
        ),
        (
            "Complete — finish all unfinished work in the project",
            lambda: _do_build(orchestrator, "complete"),
        ),
        ("Status — show a project's tasks and state", lambda: _do_status(orchestrator)),
        (
            "Report — the last build's cost and results",
            lambda: _do_report(orchestrator),
        ),
        ("Quit", None),
    ]
    try:
        while True:
            choice = _menu("What do you want to do?", [a[0] for a in actions])
            handler = actions[choice][1]
            if handler is None:
                return 0
            try:
                handler()
            except _Quit:
                return 0
            except Exception as e:  # one failed action must not crash the menu
                console.print(f"[red]  Action failed:[/] {e}")
            if not _confirm("\nDo something else?", default=True):
                return 0
    except _Quit:
        return 0
