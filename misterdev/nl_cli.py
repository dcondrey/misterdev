"""Natural-language CLI: describe what you want in English; misterdev routes it.

When ``misterdev`` is invoked with something that isn't a known subcommand, the
whole request is treated as plain English. misterdev's own model maps it to an
action, shows a preview, asks for confirmation before anything mutating, and
runs it — so there are no flags to memorize. Known subcommands still work
unchanged for power users.
"""

from typing import Any, Dict

from rich.console import Console

from misterdev.config import ConfigManager
from misterdev.llm.client import create_llm_client
from misterdev.llm.responses import extract_json_object
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)
console = Console()

# Commands the natural-language layer can resolve to. Kept in sync with the CLI.
KNOWN_COMMANDS = {"scan", "list", "status", "report", "run", "plan", "build"}
_MUTATING = {"scan", "run", "build"}

_SYSTEM = (
    "You translate a developer's plain-English request into a single misterdev "
    "action. misterdev is an autonomous build orchestrator. Reply with ONLY a "
    "JSON object, no prose."
)

_PROMPT = """Map this request to one misterdev action.

Commands:
- build: autonomously build / fix / complete a coding GOAL in a project (edits code).
- run: run a project's already-planned pending tasks.
- scan: discover and register projects under a directory.
- status: show a project's tasks and state.
- report: summarize the last build's cost and results.
- list: list registered projects.
- plan: analyze a project and propose work.

Return ONLY this JSON (omit keys you don't need):
{{"command": "<one of build|run|scan|status|report|list|plan>",
  "path": "<project or directory path, default '.'>",
  "goal": "<the plain-English goal, for build>",
  "budget": <dollars as number or null>,
  "dry_run": <true|false>,
  "parallel": <true|false>,
  "max_tasks": <int or null>}}

Request: {request}"""


def parse_intent(request: str, client) -> Dict[str, Any]:
    """One model call: request -> a structured action dict (empty on failure)."""
    text = client.generate_code(_PROMPT.format(request=request), _SYSTEM)
    obj = extract_json_object(text or "")
    return obj if isinstance(obj, dict) else {}


def _build_args(intent: Dict[str, Any]) -> str:
    """Render a build intent into the flag string ``ProjectOrchestrator.build`` parses."""
    parts = [str(intent.get("goal") or "").strip()]
    budget = intent.get("budget")
    if isinstance(budget, (int, float)):
        parts += ["--budget", str(budget)]
    if intent.get("dry_run"):
        parts.append("--dry-run")
    if intent.get("parallel"):
        parts.append("--parallel")
    mt = intent.get("max_tasks")
    if isinstance(mt, int):
        parts += ["--max-tasks", str(mt)]
    return " ".join(p for p in parts if p)


def preview(intent: Dict[str, Any]) -> str:
    """A human-readable one-liner of the resolved command."""
    cmd = intent.get("command")
    path = intent.get("path") or "."
    if cmd == "build":
        return f"build {path} {_build_args(intent)}".strip()
    if cmd == "run":
        return f"run {path}" + (" --dry-run" if intent.get("dry_run") else "")
    if cmd == "scan":
        return f"scan {path}"
    return f"{cmd} {path}".strip()


def _dispatch(intent: Dict[str, Any], orchestrator) -> int:
    cmd = intent.get("command")
    path = intent.get("path") or "."
    if cmd == "build":
        report = orchestrator.build(path, _build_args(intent))
        console.print(report)
        return 0 if orchestrator.last_build_succeeded else 1
    if cmd == "run":
        orchestrator.run_project(path, dry_run=bool(intent.get("dry_run")))
        return 0
    if cmd == "scan":
        orchestrator.scan_directory(path)
        console.print("[green]Scan complete.[/]")
        return 0
    if cmd == "status":
        console.print(orchestrator.get_project_status(path))
        return 0
    if cmd == "list":
        console.print(orchestrator.list_projects())
        return 0
    if cmd == "plan":
        orchestrator.interactive_plan(path, "")
        return 0
    if cmd == "report":
        console.print(orchestrator.get_project_status(path))
        return 0
    return 1


_MANAGEMENT_WORDS = {"scan", "list", "status", "report", "run", "plan"}
_QUERY_WORDS = {
    "what",
    "how",
    "why",
    "show",
    "get",
    "find",
    "check",
    "is",
    "are",
    "does",
    "do",
    "did",
    "has",
    "have",
    "which",
    "where",
    "when",
    "who",
}


def _fast_route(request: str) -> Dict[str, Any] | None:
    """Return a build intent without an LLM call when the request clearly
    describes coding work (first word is not a management or query word)."""
    first = request.strip().split()[0].lower() if request.strip() else ""
    if first and first not in _MANAGEMENT_WORDS and first not in _QUERY_WORDS:
        return {"command": "build", "path": ".", "goal": request.strip()}
    return None


def route(request: str, orchestrator, confirm=input) -> int:
    """Resolve a plain-English request to an action and run it.

    Returns a process exit code. ``confirm`` is injectable for testing.
    """
    intent = _fast_route(request)
    if intent is None:
        cfg = ConfigManager().load_project_config(".")
        try:
            client = create_llm_client(cfg)
        except Exception as e:
            console.print(
                "[yellow]Natural-language mode needs an LLM configured "
                f"(model + API key). {e}[/]\nRun `misterdev --help` for the flag-based CLI."
            )
            return 1
        intent = parse_intent(request, client)

    cmd = intent.get("command")
    if cmd not in KNOWN_COMMANDS:
        console.print(
            "[yellow]Couldn't map that to an action.[/] Try `misterdev --help`."
        )
        return 1

    console.print(f"[dim]→ I'll run:[/] misterdev {preview(intent)}")
    if cmd in _MUTATING:
        answer = confirm("proceed? [Y/n] ").strip().lower()
        if answer and answer not in ("y", "yes"):
            console.print("Cancelled.")
            return 0
    return _dispatch(intent, orchestrator)
