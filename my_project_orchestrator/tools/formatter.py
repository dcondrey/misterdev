from typing import Any, Tuple

from my_project_orchestrator.tools.command import CommandTool
from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)


class FormatterTool(CommandTool):
    def execute(self, project: Any, file_path: str = ".") -> Tuple[bool, str]:
        """
        Executes a code formatter.
        """
        command_template = self.config.get("command")
        if not command_template:
            return False, "No command template provided for formatter tool."

        # Only substitute a path for per-file formatters (templates containing
        # {path}). Project-wide formatters like `cargo fmt` or `ruff format .`
        # have no placeholder and must run as-is, not once per modified file.
        if "{path}" in command_template:
            command = command_template.format(path=file_path)
        else:
            command = command_template
        logger.info(f"Running formatter: {command}")
        return super().execute(project, command=command)
