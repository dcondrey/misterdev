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
        # Topography (symbol graph) is built lazily on first use, not here:
        # every CLI command registers all known projects, and eagerly scanning
        # each one's whole tree just to list/status is wasted work. The executor
        # calls initialize() (idempotent) before it needs the graph.
        self.topography = TopographyEngine(
            self.path,
            self.llm_client,
            golden_paths=get_setting(config, "orchestrator", "golden_paths"),
        )

    def _init_llm_client(self) -> BaseLLMClient:
        return create_llm_client(self.config)

    def _init_env_manager(self) -> Optional[BaseEnvironmentManager]:
        env_config = self.config.get("environment", {})
        env_type = env_config.get("type")
        if env_type == "venv":
            return VenvEnvironmentManager(env_config, self.path)
        return None
