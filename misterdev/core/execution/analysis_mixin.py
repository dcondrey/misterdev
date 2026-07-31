"""AnalysisMixin — project analysis and environment setup for ProjectOrchestrator.

Extracted from agent.py. _analyze calls self._project_file_map so the two
are kept together; _setup_env and _container_engine are co-located because
they prepare the environment that _analyze and the gate use.
"""

from typing import Optional

from misterdev.analyzers.project_analyzer import analyze_project
from misterdev.config import get_setting
from misterdev.core.execution.project import Project
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)


class AnalysisMixin:
    def _setup_env(self, project: Project) -> Optional[str]:
        """Initialize the project's env manager and return its activation prefix."""
        if project.env_manager:
            project.env_manager.setup()
            return project.env_manager.activate_command()
        return None

    def _container_engine(self, project: Project):
        """Return the project's container engine if a container environment is
        configured and an engine is available, else None (gates run locally).

        ``_setup_env`` has already called ``setup()``, so the engine is
        detected by the time gates run.
        """
        from misterdev.environments.container_env import (
            ContainerEnvironmentManager,
        )

        env = project.env_manager
        if isinstance(env, ContainerEnvironmentManager):
            return env.engine()
        return None

    def _project_file_map(self, project: Project) -> str:
        """The project's real file+symbol outline, for grounding decomposition.

        Best-effort: builds the symbol graph (idempotent) and returns its project
        outline, or "" if topography is unavailable or errors — the decomposer
        then falls back to cautious path inference rather than failing.
        """
        cached = getattr(project, "_file_map_cache", None)
        if cached is not None:
            return cached
        topo = getattr(project, "topography", None)
        if topo is None:
            return ""
        try:
            topo.initialize()
            result = topo.get_project_outline()
            project._file_map_cache = result
            return result
        except Exception as e:
            logger.warning(f"File map unavailable for decomposition (non-fatal): {e}")
            return ""

    def _analyze(self, project: Project, env_activate: Optional[str]):
        """Phase 1 analysis with config-driven commands and timeouts.

        Shared by build() and interactive_plan() so the analyzer's parameters
        (and any future config wiring) live in exactly one place.
        """
        # Build the project's symbol graph ONCE via its TopographyEngine and feed
        # the outline to the analyzer, instead of letting the source overview parse
        # a second throwaway graph. The engine's initialize() is idempotent, so the
        # later decomposition/file-map calls reuse this same graph.
        project_outline = self._project_file_map(project) or None
        return analyze_project(
            project.path,
            project.llm_client,
            build_command=project.config.get("build_command"),
            test_command=project.config.get("test_command"),
            lint_command=project.config.get("lint_command"),
            env_activate=env_activate,
            build_timeout=get_setting(project.config, "build", "build_timeout"),
            test_timeout=get_setting(project.config, "build", "test_timeout"),
            lint_timeout=get_setting(project.config, "build", "lint_timeout"),
            parallel=get_setting(project.config, "build", "parallel_analysis"),
            project_outline=project_outline,
        )
