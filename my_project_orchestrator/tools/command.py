import subprocess
from pathlib import Path
from typing import Any, Tuple

from my_project_orchestrator.tools.base_tool import BaseTool
from my_project_orchestrator.logging_setup import setup_logger
from my_project_orchestrator.utils.process import kill_process_group

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
            # start_new_session puts the command in its own process group so a
            # timeout can SIGKILL the WHOLE tree, not just the shell — otherwise a
            # backgrounded grandchild (server, compiler) is orphaned and keeps
            # running after the tool returns. errors="replace" keeps a stray
            # non-UTF8 byte from raising and being misreported as a failure.
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                kill_process_group(proc)
                # Reap the killed group, but bound the wait: a grandchild that
                # double-forked out of the group (a daemonizing server) keeps the
                # inherited stdout/stderr pipe open, so an unbounded communicate()
                # would hang the tool forever after the timeout already fired.
                try:
                    proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                error_msg = f"Command timed out after {timeout}s: {command}"
                logger.error(error_msg)
                return False, error_msg

            output = stdout
            if stderr:
                output += "\n" + stderr

            success = proc.returncode == 0
            if not success:
                logger.warning(
                    f"Command failed with exit code {proc.returncode}:\n{output}"
                )

            return success, output

        except FileNotFoundError:
            error_msg = f"Command not found: {command}"
            logger.error(error_msg)
            return False, error_msg

        except Exception as e:
            error_msg = f"Failed to execute command: {e}"
            logger.error(error_msg)
            return False, error_msg
