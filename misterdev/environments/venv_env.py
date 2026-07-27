import subprocess
from pathlib import Path

from misterdev.environments.base_env import BaseEnvironmentManager
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)


class VenvEnvironmentManager(BaseEnvironmentManager):
    def __init__(self, config: dict, project_path: str | Path):
        super().__init__(config, project_path)
        self.root_dir = self.config.get("root_dir", ".venv")
        self.venv_path = self.project_path / self.root_dir

    def setup(self) -> bool:
        """Sets up the venv if it doesn't exist."""
        if self.venv_path.exists():
            logger.info(f"Venv already exists at {self.venv_path}")
            return True

        logger.info(f"Setting up venv at {self.venv_path}")
        setup_commands = self.config.get(
            "setup_commands", [f"python -m venv {self.root_dir}"]
        )

        timeout = float(self.config.get("setup_timeout", 600))
        for cmd_template in setup_commands:
            command = cmd_template.format(root_dir=self.root_dir)
            logger.info(f"Running env setup command: {command}")
            try:
                subprocess.run(
                    command,
                    shell=True,
                    cwd=self.project_path,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                logger.error(f"Env setup command timed out after {timeout}s: {command}")
                return False
            except subprocess.CalledProcessError as e:
                logger.error(
                    f"Failed to setup environment: {e}; {e.stderr or e.stdout or ''}"
                )
                return False
        return True

    def activate_command(self) -> str:
        """Returns the activation command."""
        activate_script = self.config.get(
            "activate_script", "source {root_dir}/bin/activate"
        )
        return activate_script.format(root_dir=self.root_dir)
