import subprocess
from pathlib import Path
from typing import Any, Tuple

from my_project_orchestrator.tools.base_tool import BaseTool
from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)


class CommandTool(BaseTool):
    def execute(
        self, project: Any, command: str, cwd: str | Path = None, timeout: int = 120
    ) -> Tuple[bool, str]:
        """
        Executes a shell command.
        Returns a tuple of (success_boolean, output_or_error_string).
        """
        work_dir = cwd or project.path
        logger.info(f"Executing command: '{command}' in {work_dir}")

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=work_dir,
                capture_output=True,
                text=True,
                check=False,  # Don't raise exception on non-zero exit code
                timeout=timeout,
            )

            output = result.stdout
            if result.stderr:
                output += "\n" + result.stderr

            success = result.returncode == 0
            if not success:
                logger.warning(
                    f"Command failed with exit code {result.returncode}:\n{output}"
                )

            return success, output

        except subprocess.TimeoutExpired:
            error_msg = f"Command timed out after {timeout}s: {command}"
            logger.error(error_msg)
            return False, error_msg

        except FileNotFoundError:
            error_msg = f"Command not found: {command}"
            logger.error(error_msg)
            return False, error_msg

        except Exception as e:
            error_msg = f"Failed to execute command: {e}"
            logger.error(error_msg)
            return False, error_msg
