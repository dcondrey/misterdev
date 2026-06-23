"""Container-backed build environment (opt-in).

Selected by ``environment.type: docker`` (or ``container``). Provides a
:class:`~my_project_orchestrator.core.container.ContainerEngine` configured for
the project so gate commands run inside a pinned image. Strictly best-effort:
if no OCI engine is reachable, :meth:`engine` returns ``None`` and the caller
runs gates locally exactly as before — the environment never blocks a build.

The image comes from ``environment.image`` when set, otherwise auto-detected
from the project ``language``. ``environment.engine`` may pin a preferred engine
(podman/docker/nerdctl/colima); by default detection prefers rootless-first.
"""

from pathlib import Path
from typing import Optional

from my_project_orchestrator.core.container import (
    ContainerEngine,
    detect_engine,
    image_for_language,
)
from my_project_orchestrator.environments.base_env import BaseEnvironmentManager
from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)


class ContainerEnvironmentManager(BaseEnvironmentManager):
    def __init__(
        self,
        config: dict,
        project_path: str | Path,
        language: str = "",
        network: Optional[str] = None,
    ):
        super().__init__(config, project_path)
        self.language = language or config.get("language", "")
        self.image = config.get("image") or image_for_language(self.language)
        self.preferred_engine = config.get("engine")
        self.mount_path = config.get("mount_path", "/workspace")
        # Container egress control from governance.network ("none" disables the
        # container network). None leaves the engine default unchanged.
        self.network = network
        # Optional resource caps for the throwaway container; unset -> no flag.
        self.memory = config.get("memory")
        self.cpus = config.get("cpus")
        self.pids_limit = config.get("pids_limit")
        self._engine: Optional[ContainerEngine] = None

    def setup(self) -> bool:
        """Detect a usable OCI engine. Returns True when one is available, False
        otherwise (caller falls back to local execution). Never raises."""
        engine_name = detect_engine(self.preferred_engine)
        if engine_name is None:
            logger.info(
                "No container engine available; gate commands will run locally."
            )
            self._engine = None
            return False
        logger.info(
            f"Using container engine '{engine_name}' with image '{self.image}'."
        )
        self._engine = ContainerEngine(
            engine_name,
            self.image,
            self.project_path,
            self.mount_path,
            network=self.network,
            memory=self.memory,
            cpus=self.cpus,
            pids_limit=self.pids_limit,
        )
        return True

    def engine(self) -> Optional[ContainerEngine]:
        """The detected engine (after :meth:`setup`), or ``None`` if none."""
        return self._engine

    def activate_command(self) -> str:
        """No host-side activation: the image is the toolchain. The empty prefix
        means local fallback commands run unmodified."""
        return ""
