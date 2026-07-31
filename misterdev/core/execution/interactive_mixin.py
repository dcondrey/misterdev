"""InteractiveMixin — user-facing interaction helpers for ProjectOrchestrator.

Extracted from agent.py. Neither method calls other self methods.
"""

from rich.console import Console
from rich.prompt import Prompt

from misterdev.core.execution.project import Project
from misterdev.core.models import Task
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)
_console = Console()


class InteractiveMixin:
    def _interactive_prompt(self, task: Task, strategy: str = "iterative") -> str:
        _console.print(
            f"\n[bold cyan]Next Task:[/] [{task.id}] {task.title} ([bold magenta]{strategy.upper()}[/])"
        )
        choice = Prompt.ask("Proceed?", choices=["y", "n", "s", "q"], default="y")
        return {"y": "proceed", "q": "quit", "s": "skip", "n": "quit"}[choice]

    def _staging_hint(self, project: Project) -> str:
        """Dense-reward staging suggestion for a single complex source file.

        Uses the already-built symbol graph: when every public symbol lives in ONE
        non-test source file, that is a single-file goal — synthesize ordered
        construction->mutation->query stages so the decomposer can split it into a
        few sequential, independently-verifiable sub-tasks (raises per-attempt
        success on state-heavy files). Empty for multi-file goals or when nothing
        stages; never raises.
        """
        try:
            from misterdev.core.planning.verifier_decomposition import (
                render_stages,
                synthesize_stages,
            )

            graph = getattr(getattr(project, "topography", None), "graph", None)
            symbols = list(getattr(graph, "symbols", {}).values()) if graph else []
            src = [
                s
                for s in symbols
                if getattr(s, "file_path", "") and "test" not in s.file_path.lower()
            ]
            if len({s.file_path for s in src}) != 1:
                return ""  # staging only applies to a single-file goal
            stages = synthesize_stages(src)
            if len(stages) < 2:
                return ""
            return (
                "\n## Suggested staging (dense-reward decomposition)\n"
                "This file's public API splits into ordered, independently-"
                "verifiable stages. Prefer ONE sequential sub-task per stage, in "
                "this order (each must compile and leave the suite no worse):\n"
                f"{render_stages(stages)}\n"
            )
        except Exception as e:  # a staging hint must never break decomposition
            logger.debug(f"Staging hint skipped (non-fatal): {e}")
            return ""
