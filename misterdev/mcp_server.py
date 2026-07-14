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

from pathlib import Path
from typing import Annotated, Any, Dict, Optional

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from misterdev.agent import ProjectOrchestrator
from misterdev.core.execution.jobs import registry
from misterdev.core.planning.plan_store import load_plan, set_approval
from misterdev.core.reporting.report_view import collect

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
        title="Show the latest build report, audit trail, and model stats",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    )
)
def report(
    path: Annotated[
        str,
        Field(
            description=(
                "Absolute path to a project directory that has been built at "
                "least once. Reads only misterdev's own ``.orchestrator`` "
                "artifacts under it. Example: '/Users/me/code/my-api'."
            ),
            examples=["/Users/me/code/my-api"],
            min_length=1,
        ),
    ],
) -> Dict[str, Any]:
    """Return the latest build report, audit trail, and model performance.

    Use when: a ``build`` or ``run`` has finished (or was stopped) and you want
    the outcome — which tasks completed/failed/deferred, per-file edits, failed
    commands, governance escalations, unmet-goal gaps, token/cost totals, and
    per-model success rates. This is misterdev's read-only equivalent of asking
    "what did the last run find and do?". Do NOT use when: nothing has been
    built yet — ``latest_report`` will be null. Related: ``status`` (live task
    states), ``build``/``run`` (produce a report).

    Side effects: none — reads only the project's ``.orchestrator`` artifacts,
    calls no LLM, and returns the same result on repeated calls (idempotent).

    Returns an object with ``latest_report`` (the most recent build's structured
    summary, or null), ``audit`` (command/edit/governance counts), and
    ``models`` (per-model attempts, success rate, and average cost). Returns an
    ``error`` field instead when ``path`` is not an existing directory.
    """
    proj = Path(path).expanduser()
    if not proj.is_dir():
        return {"error": f"not a directory: {path}"}
    return collect(proj)


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


@mcp.tool(
    annotations=ToolAnnotations(
        title="Start a build in the background (returns a run_id)",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=True,
    )
)
def build_async(
    path: Annotated[
        str,
        Field(
            description=(
                "Absolute path to the project directory to build. Its git "
                "working tree must be clean. Example: '/Users/me/code/my-api'."
            ),
            examples=["/Users/me/code/my-api"],
            min_length=1,
        ),
    ],
    goal: Annotated[
        str,
        Field(
            description=(
                "What to build, or a mode word ('debug', 'complete', 'review'). "
                "Example: 'add rate limiting to the public API'."
            ),
            examples=["add rate limiting to the public API", "debug"],
            min_length=1,
        ),
    ],
    budget: Annotated[
        float,
        Field(description="Maximum US dollars to spend; must be > 0.", gt=0),
    ] = _DEFAULT_MCP_BUDGET,
    parallel: Annotated[
        bool,
        Field(description="Run independent tasks concurrently in worktrees."),
    ] = False,
    max_tasks: Annotated[
        Optional[int],
        Field(description="Cap how many tasks are planned/executed; >= 1.", ge=1),
    ] = None,
    reference_dir: Annotated[
        Optional[str],
        Field(
            description=(
                "Optional reference implementation to port from (analyzed "
                "read-only). See the synchronous ``build`` tool."
            ),
        ),
    ] = None,
) -> Dict[str, Any]:
    """Start an autonomous build in the BACKGROUND and return immediately.

    Use when: the build may run for minutes and you want to keep working — this
    returns a ``run_id`` right away instead of blocking (as the synchronous
    ``build`` does) until the run finishes. Poll ``job_status`` with the
    ``run_id`` to watch progress, ``stop_job`` to cancel, ``list_jobs`` to see
    everything running. Do NOT use for a quick preview — use ``build`` with
    ``dry_run=True``. Related: ``build`` (synchronous), ``report`` (final outcome).

    DESTRUCTIVE side effects (once running): edits files, makes git commits, and
    calls an external LLM provider. Refuses to start a second job for a project
    that already has one running (one writer per project).

    Returns ``{run_id, status}`` on success, or ``{error}`` when a job is already
    running for this project.
    """
    orch = ProjectOrchestrator()

    def _target() -> str:
        parts = [goal, "--budget", str(budget)]
        if parallel:
            parts.append("--parallel")
        if max_tasks is not None:
            parts += ["--max-tasks", str(max_tasks)]
        return orch.build(path, " ".join(parts), reference_dir=reference_dir)

    try:
        run_id = registry.start("build", path, _target, stop_hook=orch.request_stop)
    except RuntimeError as e:
        return {"error": str(e)}
    return {"run_id": run_id, "status": "running"}


@mcp.tool(
    annotations=ToolAnnotations(
        title="Start a run of planned tasks in the background (returns a run_id)",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=True,
    )
)
def run_async(
    path: Annotated[
        str,
        Field(
            description=(
                "Absolute path to a project with planned tasks (a devplan). "
                "Example: '/Users/me/code/my-api'."
            ),
            examples=["/Users/me/code/my-api"],
            min_length=1,
        ),
    ],
) -> Dict[str, Any]:
    """Start executing a project's planned tasks in the BACKGROUND.

    Use when: a devplan exists and you want it executed without blocking — like
    ``run`` but returns a ``run_id`` immediately. Poll ``job_status``, cancel
    with ``stop_job``. Do NOT use to plan from a goal — that is ``build_async``.
    Related: ``run`` (synchronous), ``build_async``.

    DESTRUCTIVE side effects (once running): edits files, makes git commits, and
    calls an external LLM provider. Refuses a second job for a project that
    already has one running.

    Returns ``{run_id, status}``, or ``{error}`` when one is already running.
    """
    orch = ProjectOrchestrator()

    def _target() -> str:
        orch.run_project(path, dry_run=False)
        return f"Ran pending tasks for {path}."

    try:
        run_id = registry.start("run", path, _target, stop_hook=orch.request_stop)
    except RuntimeError as e:
        return {"error": str(e)}
    return {"run_id": run_id, "status": "running"}


@mcp.tool(
    annotations=ToolAnnotations(
        title="Check a background job's status",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    )
)
def job_status(
    run_id: Annotated[
        str,
        Field(
            description="The run_id returned by build_async or run_async.",
            examples=["a1b2c3d4e5f6"],
            min_length=1,
        ),
    ],
) -> Dict[str, Any]:
    """Return a background job's current state.

    Use when: you started a job with ``build_async``/``run_async`` and want to
    know whether it is still ``running`` or has ``succeeded``/``failed``/
    ``stopped`` — and, when finished, its report (``result``) or ``error``.
    Read-only and idempotent. Related: ``list_jobs`` (all jobs), ``stop_job``.

    Returns the job object (run_id, kind, project_path, status, result, error,
    timestamps), or ``{error}`` when the run_id is unknown.
    """
    state = registry.status(run_id)
    if state is None:
        return {"error": f"unknown run_id: {run_id}"}
    return state


@mcp.tool(
    annotations=ToolAnnotations(
        title="Stop a running background job",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
def stop_job(
    run_id: Annotated[
        str,
        Field(
            description="The run_id of the job to stop.",
            examples=["a1b2c3d4e5f6"],
            min_length=1,
        ),
    ],
) -> Dict[str, Any]:
    """Request cooperative cancellation of a running background job.

    Use when: a ``build_async``/``run_async`` job should stop — it finishes any
    in-flight task and starts no new work, then produces a partial report. Poll
    ``job_status`` afterward to confirm it reaches ``stopped``. Idempotent:
    stopping a finished or already-stopped job is a harmless no-op. Related:
    ``job_status``, ``list_jobs``.

    Returns ``{run_id, stopping: true}`` when a running job was signalled, or
    ``{run_id, stopping: false}`` when the id is unknown or already finished.
    """
    return {"run_id": run_id, "stopping": registry.stop(run_id)}


@mcp.tool(
    annotations=ToolAnnotations(
        title="List all background jobs",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    )
)
def list_jobs() -> Dict[str, Any]:
    """List every background job this server has started and their states.

    Use when: you want an overview of all ``build_async``/``run_async`` jobs —
    running and finished — e.g. to find a lost run_id. Read-only, idempotent.
    Related: ``job_status`` (one job), ``stop_job``.

    Returns ``{jobs: [...]}``, each entry the same object ``job_status`` returns.
    """
    return {"jobs": registry.list_jobs()}


@mcp.tool(
    annotations=ToolAnnotations(
        title="Propose a ranked plan of work for approval",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    )
)
def propose_plan(
    path: Annotated[
        str,
        Field(
            description=(
                "Absolute path to the project to analyze. "
                "Example: '/Users/me/code/my-api'."
            ),
            examples=["/Users/me/code/my-api"],
            min_length=1,
        ),
    ],
    budget: Annotated[
        float,
        Field(description="Maximum US dollars to spend on analysis; > 0.", gt=0),
    ] = _DEFAULT_MCP_BUDGET,
) -> Dict[str, Any]:
    """Analyze the project and return ranked, UNAPPROVED work proposals.

    Use when: you want misterdev to recommend what to work on and let a human
    approve a subset BEFORE any code is edited — the review gate. The proposals
    are persisted, so ``get_plan`` re-reads them, ``approve_plan`` marks a
    subset, and ``execute_plan`` builds the approved ones. The codebase is
    analyzed in this process, so it never enters the client's context. Do NOT
    use to execute immediately without review — that is ``build``. Related:
    ``get_plan``, ``approve_plan``, ``execute_plan``.

    Side effects: spends LLM budget analyzing the project and writes the plan to
    ``.orchestrator/proposed_plan.json``; it edits NO source code.

    Returns ``{items: [...]}`` — each item has an id, title, work_type,
    rationale, and ``approved: false`` — or ``{error}`` on failure.
    """
    return ProjectOrchestrator().propose_plan(path, f"--budget {budget}")


@mcp.tool(
    annotations=ToolAnnotations(
        title="Read the current proposed plan and its approval state",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    )
)
def get_plan(
    path: Annotated[
        str,
        Field(
            description="Absolute path to a project that has a proposed plan.",
            examples=["/Users/me/code/my-api"],
            min_length=1,
        ),
    ],
) -> Dict[str, Any]:
    """Return the persisted proposed plan and which items are approved.

    Use when: you want to review the proposals from ``propose_plan`` (and see
    what has been approved so far) before approving or executing. Read-only and
    idempotent. Related: ``propose_plan``, ``approve_plan``, ``execute_plan``.

    Returns ``{items: [...]}`` (empty ``items`` when no plan has been proposed).
    """
    plan = load_plan(path)
    return {"items": plan or []}


@mcp.tool(
    annotations=ToolAnnotations(
        title="Approve or reject items in the proposed plan",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
def approve_plan(
    path: Annotated[
        str,
        Field(
            description="Absolute path to a project that has a proposed plan.",
            examples=["/Users/me/code/my-api"],
            min_length=1,
        ),
    ],
    approve_all: Annotated[
        bool,
        Field(description="Approve every item in the plan."),
    ] = False,
    approve_ids: Annotated[
        Optional[list[str]],
        Field(
            description="Ids to approve (as shown by get_plan, e.g. 'P-001').",
            examples=[["P-001", "P-003"]],
        ),
    ] = None,
    reject_ids: Annotated[
        Optional[list[str]],
        Field(
            description="Ids to un-approve. An id in both lists is rejected.",
            examples=[["P-002"]],
        ),
    ] = None,
) -> Dict[str, Any]:
    """Set the approval flags on the proposed plan, then persist.

    Use when: after ``propose_plan``/``get_plan`` you want to mark which items
    should actually run. ``approve_all`` approves everything; otherwise
    ``approve_ids`` are approved and ``reject_ids`` un-approved (reject wins a
    tie — the safer default). Idempotent. Then call ``execute_plan``. Related:
    ``propose_plan``, ``get_plan``, ``execute_plan``.

    Side effects: rewrites ``.orchestrator/proposed_plan.json``; edits no code.

    Returns ``{items: [...]}`` with updated flags, or ``{error}`` when no plan
    exists to approve.
    """
    plan = set_approval(
        path,
        item_ids=approve_ids,
        approve_all=approve_all,
        reject_ids=reject_ids,
    )
    if plan is None:
        return {"error": "no proposed plan for this project; call propose_plan first"}
    return {"items": plan}


@mcp.tool(
    annotations=ToolAnnotations(
        title="Build the approved items from the proposed plan",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=True,
    )
)
def execute_plan(
    path: Annotated[
        str,
        Field(
            description="Absolute path to a project with an approved plan.",
            examples=["/Users/me/code/my-api"],
            min_length=1,
        ),
    ],
    budget: Annotated[
        float,
        Field(description="Maximum US dollars to spend; > 0.", gt=0),
    ] = _DEFAULT_MCP_BUDGET,
) -> str:
    """Execute the APPROVED items from a previously proposed plan.

    Use when: ``propose_plan`` + ``approve_plan`` have selected the work and you
    want it built — this composes a goal from the approved items and runs the
    normal build pipeline (decompose, edit, verify, revert regressions). Do NOT
    use before approving anything (it returns a no-op message). Related:
    ``propose_plan``, ``approve_plan``, ``build``.

    DESTRUCTIVE side effects: edits files, makes git commits, and calls an
    external LLM provider. Refuses a dirty working tree (like ``build``).

    Returns a compact build report, or a message when nothing is approved.
    """
    return ProjectOrchestrator().execute_plan(path, f"--budget {budget}")


def main() -> None:
    """Console entry point: serve misterdev over stdio MCP."""
    mcp.run()
