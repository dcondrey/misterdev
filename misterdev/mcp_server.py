"""misterdev as an MCP server: drive the orchestrator from any MCP client.

This is a THIN adapter over the same ``ProjectOrchestrator`` the CLI uses. The
heavy work — reading the codebase, symbol-graph context management, multi-step
reasoning, model selection, budget — all runs IN THIS PROCESS with misterdev's
own LLM key. The client only sends a short instruction and receives a short
summary, so the codebase never enters the client's context window and the
context-scaling misterdev exists to provide is fully preserved.

Run it with the ``misterdev-mcp`` entry point (needs the ``mcp`` extra:
``pip install 'misterdev[mcp]'``). Mutating tools (scan/build/run) modify the
target repo; ``build``/``run`` refuse a dirty working tree and carry a
conservative default budget.
"""

from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from misterdev.agent import ProjectOrchestrator

# Conservative default $ ceiling for an AI-client-triggered build (the CLI
# default is higher). The client can raise it explicitly per call.
_DEFAULT_MCP_BUDGET = 10.0

mcp = FastMCP("misterdev")


@mcp.tool()
def scan(directory: str) -> str:
    """Discover misterdev projects under a directory and register them.

    Mutating: writes to the project registry. Returns a short confirmation.
    """
    ProjectOrchestrator().scan_directory(directory)
    return f"Scanned and registered projects under: {directory}"


@mcp.tool()
def list_projects() -> Dict[str, Any]:
    """List the projects misterdev currently knows about (read-only)."""
    return ProjectOrchestrator().list_projects()


@mcp.tool()
def status(path: str) -> Dict[str, Any]:
    """Show a project's tasks and their state (read-only)."""
    return ProjectOrchestrator().get_project_status(path)


@mcp.tool()
def build(
    path: str,
    goal: str,
    budget: float = _DEFAULT_MCP_BUDGET,
    dry_run: bool = False,
    parallel: bool = False,
    max_tasks: Optional[int] = None,
) -> str:
    """Run the autonomous build/debug/complete loop on a project.

    ``goal`` is plain English (or a mode word: debug, complete, review). MUTATING
    — it edits the repo. Refuses a dirty working tree. Use ``dry_run=True`` to
    plan without executing. ``budget`` caps dollar spend; ``max_tasks`` caps how
    many tasks are planned/run. Returns a compact report summary.
    """
    orch = ProjectOrchestrator()
    parts = [goal, "--budget", str(budget)]
    if dry_run:
        parts.append("--dry-run")
    if parallel:
        parts.append("--parallel")
    if max_tasks is not None:
        parts += ["--max-tasks", str(max_tasks)]
    report = orch.build(path, " ".join(parts))
    outcome = "succeeded" if orch.last_build_succeeded else "did not fully succeed"
    return f"Build {outcome}.\n\n{report}"


@mcp.tool()
def run(path: str, task_id: Optional[str] = None, dry_run: bool = False) -> str:
    """Run a project's pending tasks (or one ``task_id``). MUTATING.

    Use ``dry_run=True`` to preview without executing. Returns a summary.
    """
    orch = ProjectOrchestrator()
    if task_id:
        orch.run_task(path, task_id)
        return f"Ran task {task_id} for {path}."
    orch.run_project(path, dry_run=dry_run)
    verb = "Previewed" if dry_run else "Ran"
    return f"{verb} pending tasks for {path}."


def main() -> None:
    """Console entry point: serve misterdev over stdio MCP."""
    mcp.run()
