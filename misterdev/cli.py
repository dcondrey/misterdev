import argparse
import sys
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from misterdev import __version__
from misterdev.agent import ProjectOrchestrator
from misterdev.logging_setup import setup_logger

logger = setup_logger("cli")
console = Console()


def _print_report(project_path: str) -> None:
    """Render the aggregated cost/model/audit summary for a project."""
    from misterdev.core.reporting.report_view import collect

    data = collect(project_path)
    rpt = data["latest_report"]
    if rpt:
        cost = rpt.get("llm_cost", 0.0) or 0.0
        console.print(
            f"[bold]Latest build[/] — {rpt.get('mode', '?')} on "
            f"{rpt.get('project', '?')}  "
            f"[{'green' if rpt.get('validation_passed') else 'red'}]"
            f"{'PASSED' if rpt.get('validation_passed') else 'not green'}[/]"
        )
        console.print(
            f"  ${cost:.4f} over {rpt.get('llm_calls', 0)} call(s), "
            f"{rpt.get('llm_tokens', 0):,} tokens · "
            f"completed {len(rpt.get('completed', []))}, "
            f"failed {len(rpt.get('failed', []))}, "
            f"deferred {len(rpt.get('deferred', []))}"
        )
        p_in = rpt.get("llm_prompt_tokens", 0) or 0
        p_out = rpt.get("llm_completion_tokens", 0) or 0
        p_cache = rpt.get("llm_cache_read_tokens", 0) or 0
        if p_in or p_out:
            cached = f", {p_cache:,} cached" if p_cache else ""
            console.print(
                f"  [dim]tokens: {p_in:,} input / {p_out:,} output{cached}[/]"
            )
    else:
        console.print("[dim]No saved build report found.[/]")

    models = data["models"]
    if models:
        table = Table(title="\nModel performance (from the ledger)")
        table.add_column("Model", style="cyan")
        table.add_column("Attempts", justify="right")
        table.add_column("Pass rate", justify="right")
        table.add_column("First-try", justify="right")
        table.add_column("Avg $/success", justify="right")
        for m in models:
            table.add_row(
                m["model"],
                str(m["attempts"]),
                f"{m['success_rate'] * 100:.0f}%",
                f"{m['first_try_rate'] * 100:.0f}%",
                f"${m['avg_cost']:.4f}",
            )
        console.print(table)
    else:
        console.print("[dim]No model ledger yet.[/]")

    audit = data["audit"]
    if audit["total_events"]:
        cmds = audit["commands"]
        gov = audit["governance"]
        console.print(
            f"\n[bold]Audit trail[/] — {audit['total_events']} event(s): "
            f"{cmds['ok']} command(s) ok / {cmds['failed']} failed, "
            f"{audit['edits']['total']} edit(s)"
            + (
                f", {gov['escalated']} governance escalation(s)"
                if gov["escalated"]
                else ""
            )
        )
        top = sorted(
            audit["edits"]["by_file"].items(), key=lambda kv: kv[1], reverse=True
        )[:5]
        for path, n in top:
            console.print(f"  [dim]{n}×[/] {path}")
    else:
        console.print("[dim]No audit trail yet.[/]")


def main():
    # Natural-language mode: when the first argument isn't a known subcommand
    # (and isn't a flag), treat the whole line as plain English and let
    # misterdev's model route it — no flags to memorize. Known subcommands fall
    # through to the flag-based parser below, so power users are unaffected.
    import sys

    _known = {
        "scan",
        "list",
        "status",
        "report",
        "run",
        "plan",
        "build",
        "interactive",
        "i",
    }
    argv = sys.argv[1:]
    if argv and not argv[0].startswith("-") and argv[0] not in _known:
        from misterdev.nl_cli import route

        sys.exit(route(" ".join(argv), ProjectOrchestrator()))

    parser = argparse.ArgumentParser(
        description="misterdev — autonomous build orchestrator"
    )
    parser.add_argument(
        "--version", action="version", version=f"misterdev {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 'scan' command
    scan_parser = subparsers.add_parser("scan", help="Scan a directory for projects")
    scan_parser.add_argument("directory", type=str, help="Directory to scan")

    # 'list' command
    subparsers.add_parser("list", help="List all registered projects")

    # 'status' command
    status_parser = subparsers.add_parser("status", help="Show status of a project")
    status_parser.add_argument(
        "project_path", type=str, nargs="?", default=".", help="Path to the project"
    )

    # 'report' command (aggregated cost/audit/model view)
    report_parser = subparsers.add_parser(
        "report",
        help="Summarize cost, model performance, and the audit trail for a project",
    )
    report_parser.add_argument(
        "project_path", type=str, nargs="?", default=".", help="Path to the project"
    )

    # 'run' command (legacy)
    run_parser = subparsers.add_parser("run", help="Run tasks for a project")
    run_parser.add_argument(
        "project_path",
        type=str,
        nargs="?",
        default=".",
        help="Path to the project (defaults to current dir)",
    )
    run_parser.add_argument("--task", type=str, help="Specific task ID to run")
    run_parser.add_argument(
        "--tasks",
        type=str,
        metavar="FILE",
        help="Path to an external task-list file (JSON/YAML/Markdown/text, any "
        "layout — may live in another repo). Overrides the devplan directory.",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show execution plan without running tasks",
    )
    run_parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip pre-flight devplan validation",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all tasks, ignoring the completion cache",
    )
    run_parser.add_argument(
        "--status",
        action="store_true",
        help="Show which tasks would run vs skip, then exit",
    )
    run_parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Max dollar ceiling for this run (tighter of this and build.budget "
        "wins). Omit to use the project.yaml build.budget.",
    )
    run_parser.add_argument(
        "--proceed",
        action="store_true",
        help="Skip the requirements-preflight stop: run now and park any missing "
        "inputs instead of stopping to gather foundational ones first.",
    )

    # 'interactive' / 'i' — guided menu (also the no-argument default)
    subparsers.add_parser(
        "interactive",
        aliases=["i"],
        help="Guided menu — pick an action, answer a couple of prompts (no flags)",
    )

    # 'plan' command (interactive: analyze -> recommend -> compose -> confirm)
    plan_parser = subparsers.add_parser(
        "plan",
        help="Analyze the project, recommend work, and compose a plan interactively",
    )
    plan_parser.add_argument(
        "project_path", type=str, nargs="?", default=".", help="Path to the project"
    )
    plan_parser.add_argument(
        "--budget", type=float, default=100.0, help="Max dollar budget"
    )
    plan_parser.add_argument(
        "--no-rollback",
        action="store_true",
        help="Disable integration-gate revert of regressing tasks",
    )

    # 'build' command (6-phase workflow from /build skill)
    build_parser = subparsers.add_parser(
        "build", help="Autonomous build/debug/complete workflow"
    )
    build_parser.add_argument(
        "project_path", type=str, nargs="?", default=".", help="Path to the project"
    )
    build_parser.add_argument(
        "prompt",
        nargs="*",
        default=[],
        help="Mode or description (debug, complete, review, new <desc>, or free text)",
    )
    build_parser.add_argument(
        "--budget", type=float, default=100.0, help="Max dollar budget"
    )
    build_parser.add_argument(
        "--commit", action="store_true", help="Commit after each completed task"
    )
    build_parser.add_argument(
        "--no-verify", action="store_true", help="Skip final validation phase"
    )
    build_parser.add_argument(
        "--no-suggest", action="store_true", help="Skip suggest scan"
    )
    build_parser.add_argument(
        "--dry-run", action="store_true", help="Plan only, show tasks without executing"
    )
    build_parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Wait for user confirmation between tasks",
    )
    build_parser.add_argument(
        "--parallel", action="store_true", help="Execute independent tasks concurrently"
    )
    build_parser.add_argument(
        "--no-rollback",
        action="store_true",
        help="Disable auto-bisect/revert of a regressing task on gate failure",
    )
    build_parser.add_argument(
        "--focus", type=str, help="Restrict work to specific area"
    )
    build_parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow building on a working tree with uncommitted changes",
    )
    build_parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Cap the number of tasks this run will plan/execute (bounds cost)",
    )

    args = parser.parse_args()

    orchestrator = ProjectOrchestrator()

    if args.command == "scan":
        orchestrator.scan_directory(args.directory)
        console.print("[green]Scan complete.[/]")
    elif args.command == "list":
        projects = orchestrator.list_projects()
        if not projects:
            console.print("[dim]No projects registered.[/]")
        else:
            table = Table(title="Registered Projects")
            table.add_column("Name", style="cyan")
            table.add_column("Path", style="dim")
            for path, info in projects.items():
                table.add_row(info["name"], path)
            console.print(table)
    elif args.command == "status":
        status = orchestrator.get_project_status(args.project_path)
        if "error" in status:
            console.print(f"[red]Error:[/] {status['error']}")
        else:
            console.print(f"[bold]{status['name']}[/]  {status['path']}")
            if status["description"]:
                console.print(f"[dim]{status['description']}[/]")
            if not status["tasks"]:
                console.print("\n[dim]No tasks found.[/]")
            else:
                table = Table(title="Tasks")
                table.add_column("ID", style="cyan")
                table.add_column("Status")
                table.add_column("Description", max_width=60)
                for task in status["tasks"]:
                    style = {
                        "completed": "green",
                        "failed": "red",
                        "in_progress": "yellow",
                    }.get(task["status"], "dim")
                    table.add_row(
                        task["id"], f"[{style}]{task['status']}[/]", task["description"]
                    )
                console.print(table)
    elif args.command == "report":
        _print_report(args.project_path)
    elif args.command == "run":
        if args.task:
            logger.info(
                f"Running specific task {args.task} in project {args.project_path}"
            )
            orchestrator.run_task(args.project_path, args.task)
        else:
            logger.info(f"Running pending tasks for project {args.project_path}")
            orchestrator.run_project(
                args.project_path,
                dry_run=args.dry_run,
                skip_preflight=args.skip_preflight,
                force=args.force,
                status=args.status,
                tasklist=args.tasks,
                budget=args.budget,
                proceed=args.proceed,
            )
    elif args.command == "build":
        build_args = list(args.prompt)
        if args.budget != 100.0:
            build_args.extend(["--budget", str(args.budget)])
        if args.commit:
            build_args.append("--commit")
        if args.no_verify:
            build_args.append("--no-verify")
        if args.no_suggest:
            build_args.append("--no-suggest")
        if args.dry_run:
            build_args.append("--dry-run")
        if args.interactive:
            build_args.append("--interactive")
        if args.parallel:
            build_args.append("--parallel")
        if args.no_rollback:
            build_args.append("--no-rollback")
        if args.allow_dirty:
            build_args.append("--allow-dirty")
        if args.focus:
            build_args.extend(["--focus", args.focus])
        if args.max_tasks is not None:
            build_args.extend(["--max-tasks", str(args.max_tasks)])

        report = orchestrator.build(args.project_path, " ".join(build_args))
        console.print("\n")
        if orchestrator.last_build_succeeded:
            console.print(
                Panel(
                    Markdown(report),
                    title="[bold green]Build Complete[/bold green]",
                    expand=False,
                )
            )
        else:
            console.print(
                Panel(
                    Markdown(report),
                    title="[bold red]Build Failed Validation[/bold red]",
                    expand=False,
                )
            )
            sys.exit(1)
    elif args.command in ("interactive", "i") or args.command is None:
        # Plain `misterdev` (no subcommand) launches the guided menu.
        from misterdev.interactive import run_interactive

        sys.exit(run_interactive(orchestrator))
    elif args.command == "plan":
        path = getattr(args, "project_path", ".")
        plan_args = []
        if getattr(args, "budget", 100.0) != 100.0:
            plan_args.extend(["--budget", str(args.budget)])
        if getattr(args, "no_rollback", False):
            plan_args.append("--no-rollback")
        report = orchestrator.interactive_plan(path, " ".join(plan_args))
        console.print("\n")
        title = (
            "[bold green]Build Complete[/bold green]"
            if orchestrator.last_build_succeeded
            else "[bold yellow]Planning Result[/bold yellow]"
        )
        console.print(Panel(Markdown(report), title=title, expand=False))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/]")
        sys.exit(130)
