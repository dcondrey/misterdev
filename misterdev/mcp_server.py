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
        title="List every project in the registry",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    )
)
def list_projects() -> Dict[str, Any]:
    """List every project misterdev currently knows about.

    Use when: you need to discover which projects are registered before calling
    ``status``, ``build``, or ``run`` on a specific one. Do NOT use when: the
    project isn't registered yet — call ``scan`` first to register it. Related:
    ``scan`` (register projects), ``status`` (inspect one project). Takes no
    parameters.

    Side effects: none — read-only, calls no LLM, and returns the same result on
    repeated calls (idempotent).

    Returns a mapping of project id to an object with its registered ``path`` and
    ``name``; an empty mapping when nothing is registered.
    """
    return ProjectOrchestrator().list_projects()


@mcp.tool(
    annotations=ToolAnnotations(
        title="Show a project's tasks and their state",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    )
)
def status(
    path: Annotated[
        str,
        Field(
            description=(
                "Absolute path to a registered project's directory (the one "
                "containing its project.yaml). Must be a directory that has been "
                "registered via ``scan``. Example: '/Users/me/code/my-api'."
            ),
            examples=["/Users/me/code/my-api", "/workspace/repos/service"],
            min_length=1,
        ),
    ],
) -> Dict[str, Any]:
    """Show a project's tasks and their current state.

    Use when: you want to inspect what work exists and how far it has progressed,
    e.g. before deciding whether to ``run`` pending tasks or ``build`` new work.
    Do NOT use when: the project isn't registered — call ``scan`` first, or
    ``list_projects`` to find the right path. Related: ``list_projects``,
    ``run``, ``build``.

    Side effects: none — read-only, calls no LLM, idempotent.

    Returns the project's tasks, each with its id, title, and status.
    """
    return ProjectOrchestrator().get_project_status(path)


@mcp.tool(
    annotations=ToolAnnotations(
        title="Scan a directory and register the projects found",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
def scan(
    directory: Annotated[
        str,
        Field(
            description=(
                "Absolute path to an existing, readable directory to search "
                "recursively for misterdev projects (directories containing a "
                "project.yaml). Must be a directory, not a file. "
                "Example: '/Users/me/code'."
            ),
            examples=["/Users/me/code", "/workspace/repos"],
            min_length=1,
        ),
    ],
) -> str:
    """Discover misterdev projects under a directory and add them to the registry.

    Use when: you have projects on disk that misterdev does not know about yet,
    before calling ``status``, ``build``, or ``run`` on them. Do NOT use when:
    the projects are already registered (call ``list_projects`` to check) — a
    re-scan is harmless but redundant. Related: ``list_projects`` (see what is
    registered), ``status`` (inspect a registered project).

    Side effects: writes only to misterdev's project registry — it never reads,
    edits, or executes any project code, and re-scanning the same directory is
    idempotent (no duplicates).

    Returns a short confirmation string naming the directory scanned.
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
        str,
        Field(
            description=(
                "Absolute path to the project directory to build (containing "
                "project.yaml). Its git working tree must be clean. "
                "Example: '/Users/me/code/my-api'."
            ),
            examples=["/Users/me/code/my-api"],
            min_length=1,
        ),
    ],
    goal: Annotated[
        str,
        Field(
            description=(
                "Plain-English description of what to build, or a mode word: "
                "'debug' (fix what's broken), 'complete' (finish unfinished work), "
                "or 'review'. Example: 'add rate limiting to the public API'."
            ),
            examples=["add rate limiting to the public API", "debug", "complete"],
            min_length=1,
        ),
    ],
    budget: Annotated[
        float,
        Field(
            description=(
                "Maximum US dollars to spend on this run. Must be > 0; the run "
                "halts when reached. Example: 5.0."
            ),
            examples=[5.0, 25.0],
            gt=0,
        ),
    ] = _DEFAULT_MCP_BUDGET,
    dry_run: Annotated[
        bool,
        Field(
            description=(
                "When true, plan and preview the tasks without editing any code "
                "or spending beyond planning. Example: true."
            ),
            examples=[False, True],
        ),
    ] = False,
    parallel: Annotated[
        bool,
        Field(
            description=(
                "When true, run independent tasks concurrently in isolated git "
                "worktrees. Example: false."
            ),
            examples=[False, True],
        ),
    ] = False,
    max_tasks: Annotated[
        Optional[int],
        Field(
            description=(
                "Cap how many tasks are planned/executed (bounds cost and scope). "
                "Must be >= 1; omit for no cap. Example: 5."
            ),
            examples=[5, 10],
            ge=1,
        ),
    ] = None,
    reference_dir: Annotated[
        Optional[str],
        Field(
            description=(
                "Absolute path to a reference implementation to port from (often "
                "in another language). Its module/symbol map is extracted "
                "READ-ONLY and given to the planner so the build reproduces the "
                "reference's design idiomatically. Omit when not porting. "
                "Example: '/Users/me/code/donor-impl'."
            ),
            examples=["/Users/me/code/donor-impl"],
        ),
    ] = None,
) -> str:
    """Autonomously plan AND execute a goal in a project, from scratch.

    Use when: you have a goal but no existing task plan — ``build`` analyzes the
    project, decomposes ``goal`` into tasks, edits the code, and verifies each
    change through build/test/lint/typecheck gates, reverting anything that
    regresses. Do NOT use when: a task plan already exists and you just want to
    execute it (use ``run``), or the working tree is dirty (commit/stash first).
    Related: ``run`` (execute an existing plan), ``status`` (inspect tasks).
    Pass ``reference_dir`` to port from an existing implementation: its
    module/symbol map is extracted read-only and guides the plan.

    DESTRUCTIVE side effects: edits files and makes git commits, and calls an
    external LLM provider (open-world, non-idempotent). It refuses to run on a
    dirty working tree. ``dry_run=True`` previews without changing anything;
    ``budget`` caps spend; ``max_tasks`` caps scope.

    Returns a compact text report: what was done, per-gate results, and cost.
    """
    orch = ProjectOrchestrator()
    parts = [goal, "--budget", str(budget)]
    if dry_run:
        parts.append("--dry-run")
    if parallel:
        parts.append("--parallel")
    if max_tasks is not None:
        parts += ["--max-tasks", str(max_tasks)]
    report = orch.build(path, " ".join(parts), reference_dir=reference_dir)
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
    path: Annotated[
        str,
        Field(
            description=(
                "Absolute path to a registered project's directory that already "
                "has planned tasks (a devplan). Example: '/Users/me/code/my-api'."
            ),
            examples=["/Users/me/code/my-api"],
            min_length=1,
        ),
    ],
    task_id: Annotated[
        Optional[str],
        Field(
            description=(
                "Run only this single task id (as shown by ``status``); omit to "
                "run all pending tasks in dependency order. Example: '010-add-auth'."
            ),
            examples=["010-add-auth", "T-003"],
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        Field(
            description=(
                "When true, preview the tasks that would run without executing "
                "or editing anything. Example: true."
            ),
            examples=[False, True],
        ),
    ] = False,
) -> str:
    """Execute a project's ALREADY-PLANNED pending tasks (a devplan).

    Use when: tasks already exist (from a prior ``plan`` or a devplan directory)
    and you want to execute them. Do NOT use when: no plan exists and you are
    starting from a goal — that is ``build``'s job (it analyzes and decomposes).
    Related: ``build`` (plan + execute a goal), ``status`` (see the task ids).
    Pass ``task_id`` to run a single task.

    DESTRUCTIVE side effects: edits files and makes git commits, and calls an
    external LLM provider (open-world, non-idempotent). ``dry_run=True`` previews
    without changing anything.

    Returns a short text summary of what was run or previewed.
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
