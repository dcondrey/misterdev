"""misterdev as an MCP server: drive the orchestrator from any MCP client.

This is a THIN adapter over the same ``ProjectOrchestrator`` the CLI uses. The
heavy work — reading the codebase, symbol-graph context management, multi-step
reasoning, model selection, budget — all runs IN THIS PROCESS with misterdev's
own LLM key. The client only sends a short instruction and receives a short
summary, so the codebase never enters the client's context window and the
context-scaling misterdev exists to provide is fully preserved.

Run it with the ``misterdev-mcp`` entry point (needs the ``mcp`` extra:
``pip install 'misterdev[mcp]'``). Tool definitions carry per-parameter
descriptions and honest behavioral annotations (read-only vs. destructive,
idempotent, open-world) so an AI agent can pick and call them correctly.
"""

from typing import Annotated, Any, Dict, Optional

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from misterdev.agent import ProjectOrchestrator

# Conservative default $ ceiling for an AI-client-triggered build (the CLI
# default is higher). The client can raise it explicitly per call.
_DEFAULT_MCP_BUDGET = 10.0

mcp = FastMCP("misterdev")


@mcp.tool(
    annotations=ToolAnnotations(
        title="List registered projects",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    )
)
def list_projects() -> Dict[str, Any]:
    """List every project misterdev currently knows about.

    Use this first to see what is registered before calling ``status`` or
    ``build`` on a specific one. Read-only — nothing is changed.

    Returns a mapping of project id to its registered path and name.
    """
    return ProjectOrchestrator().list_projects()


@mcp.tool(
    annotations=ToolAnnotations(
        title="Project status",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    )
)
def status(
    path: Annotated[
        str,
        Field(
            description="Path to the project (its directory, containing project.yaml)."
        ),
    ],
) -> Dict[str, Any]:
    """Show a project's tasks and their current state.

    Use to inspect what work exists and how far it has progressed, e.g. before
    deciding whether to ``run`` pending tasks or ``build`` something new.
    Read-only — it does not run anything.

    Returns the project's tasks with their statuses.
    """
    return ProjectOrchestrator().get_project_status(path)


@mcp.tool(
    annotations=ToolAnnotations(
        title="Scan for projects",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
def scan(
    directory: Annotated[
        str,
        Field(description="Directory to search recursively for projects to register."),
    ],
) -> str:
    """Discover misterdev projects under a directory and register them.

    Use once to make projects known to misterdev before inspecting or building
    them. It writes to the project registry but does not touch project code, and
    re-scanning the same directory is idempotent.

    Returns a short confirmation string.
    """
    ProjectOrchestrator().scan_directory(directory)
    return f"Scanned and registered projects under: {directory}"


@mcp.tool(
    annotations=ToolAnnotations(
        title="Autonomous build",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=True,
    )
)
def build(
    path: Annotated[
        str, Field(description="Path to the project to build (its directory).")
    ],
    goal: Annotated[
        str,
        Field(
            description="Plain-English goal, or a mode word: 'debug', 'complete', or 'review'."
        ),
    ],
    budget: Annotated[
        float,
        Field(description="Maximum dollars to spend on this run.", gt=0),
    ] = _DEFAULT_MCP_BUDGET,
    dry_run: Annotated[
        bool,
        Field(description="Plan and preview tasks without editing any code."),
    ] = False,
    parallel: Annotated[
        bool,
        Field(
            description="Run independent tasks concurrently in isolated git worktrees."
        ),
    ] = False,
    max_tasks: Annotated[
        Optional[int],
        Field(description="Cap how many tasks are planned/executed (bounds cost)."),
    ] = None,
) -> str:
    """Autonomously plan AND execute a goal in a project, from scratch.

    This is the main tool. Unlike ``run`` (which executes an existing plan),
    ``build`` analyzes the project, decomposes ``goal`` into tasks, edits the
    code, and verifies each change through build/test/lint/typecheck gates,
    reverting anything that regresses.

    DESTRUCTIVE: it edits files and makes git commits. It refuses to run on a
    dirty working tree (commit or stash first). Use ``dry_run=True`` to preview
    the plan without changing anything. ``budget`` caps spend; ``max_tasks``
    caps scope. It calls an external LLM provider (open-world).

    Returns a compact report: what was done, gate results, and cost.
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


@mcp.tool(
    annotations=ToolAnnotations(
        title="Run planned tasks",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=True,
    )
)
def run(
    path: Annotated[str, Field(description="Path to the project.")],
    task_id: Annotated[
        Optional[str],
        Field(description="Run only this task id; omit to run all pending tasks."),
    ] = None,
    dry_run: Annotated[
        bool, Field(description="Preview the tasks without executing them.")
    ] = False,
) -> str:
    """Execute a project's ALREADY-PLANNED pending tasks (a devplan).

    Use this when tasks already exist (from a prior ``plan`` or devplan) and you
    just want to run them — it does NOT analyze or decompose a goal; that is
    ``build``'s job. Pass ``task_id`` to run a single task.

    DESTRUCTIVE: it edits files and commits. Use ``dry_run=True`` to preview.
    Calls an external LLM provider (open-world). Returns a summary.
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
