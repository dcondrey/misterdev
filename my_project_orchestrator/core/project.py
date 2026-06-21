from pathlib import Path
from typing import Optional

from my_project_orchestrator.config import get_setting
from my_project_orchestrator.llm.client import BaseLLMClient, create_llm_client
from my_project_orchestrator.environments.base_env import BaseEnvironmentManager
from my_project_orchestrator.environments.venv_env import VenvEnvironmentManager
from my_project_orchestrator.core.task import TaskManager
from my_project_orchestrator.core.topography import TopographyEngine
from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)


class ToolManager:
    """Manages initialization and execution of tools."""

    def __init__(self, tools_config: list):
        self.tools = {}
        for tc in tools_config:
            # We would typically use a factory here based on tc['type']
            from my_project_orchestrator.tools.command import CommandTool
            from my_project_orchestrator.tools.formatter import FormatterTool
            from my_project_orchestrator.tools.git_tool import GitTool
            from my_project_orchestrator.tools.file_io import FileIOTool

            tool_type = tc.get("type")
            if tool_type == "formatter":
                tool = FormatterTool(tc)
            elif tool_type == "git":
                tool = GitTool(tc)
            elif tool_type == "file_io":
                tool = FileIOTool(tc)
            elif tool_type in ["test_runner", "command"]:
                tool = CommandTool(tc)
            else:
                tool = CommandTool(tc)  # Fallback
            self.tools[tool.name] = tool

    def get_tool(self, name: str):
        return self.tools.get(name)


class Project:
    """Represents an active project with all its dependencies initialized."""

    def __init__(self, path: str | Path, config: dict):
        self.path = Path(path)
        self.config = config

        self.name = config.get("name", self.path.name)
        self.description = config.get("description", "")

        self.llm_client = self._init_llm_client()
        self.env_manager = self._init_env_manager()
        self.tool_manager = ToolManager(config.get("tools", []))
        self.task_manager = TaskManager(self)
        # Model ledger/selector are built lazily on first use: they touch the
        # .orchestrator dir and only matter when dynamic_selection is enabled.
        self._model_ledger = None
        self._model_selector = None
        self._llm_cache = None
        # Topography (symbol graph) is built lazily on first use, not here:
        # every CLI command registers all known projects, and eagerly scanning
        # each one's whole tree just to list/status is wasted work. The executor
        # calls initialize() (idempotent) before it needs the graph.
        self.topography = TopographyEngine(
            self.path,
            self.llm_client,
            golden_paths=get_setting(config, "orchestrator", "golden_paths"),
        )

    @property
    def model_ledger(self):
        """Persistent per-model performance store (lazy, file-backed)."""
        if self._model_ledger is None:
            from my_project_orchestrator.core.model_ledger import ModelLedger

            self._model_ledger = ModelLedger(
                self.path / ".orchestrator" / "model_stats.json"
            )
        return self._model_ledger

    @property
    def model_selector(self):
        """Ledger-driven model selection policy (lazy)."""
        if self._model_selector is None:
            from my_project_orchestrator.core.model_selector import ModelSelector

            self._model_selector = ModelSelector(
                self.config, self.model_ledger, free_models=self._harvest_free_models()
            )
        return self._model_selector

    @property
    def llm_cache(self):
        """Response memoization store, or None when caching is disabled."""
        if self._llm_cache is None and get_setting(self.config, "llm", "cache"):
            from my_project_orchestrator.core.llm_cache import LLMCache

            self._llm_cache = LLMCache(self.path / ".orchestrator" / "llm_cache")
        return self._llm_cache

    def _harvest_free_models(self) -> list:
        """Current free OpenRouter models when use_free_models is enabled."""
        if not get_setting(self.config, "llm", "use_free_models"):
            return []
        import time

        from my_project_orchestrator.core.free_models import FreeModelCache

        cache = FreeModelCache(self.path / ".orchestrator" / "free_models.json")
        try:
            return cache.get(time.time())
        except Exception as e:
            logger.warning(f"Free-model harvest skipped: {e}")
            return []

    def _init_llm_client(self) -> BaseLLMClient:
        return create_llm_client(self.config)

    def _init_env_manager(self) -> Optional[BaseEnvironmentManager]:
        env_config = self.config.get("environment", {})
        env_type = env_config.get("type")
        if env_type == "venv":
            return VenvEnvironmentManager(env_config, self.path)
        return None
