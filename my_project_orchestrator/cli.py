import argparse
import sys
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from my_project_orchestrator.agent import ProjectOrchestrator
from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger("cli")
console = Console()


def main():
    parser = argparse.ArgumentParser(description="Project Orchestrator CLI")
    parser.add_argument(
        "--version", action="version", version="project-orchestrator 0.1.0"
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
    elif args.command == "plan" or args.command is None:
        # Plain `project-orchestrator` (no subcommand) launches interactive planning.
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
