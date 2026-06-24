"""Engine-agnostic OCI container execution substrate (opt-in).

Runs gate commands (build/lint/test/golden) inside a throwaway container so a
build is exercised against a pinned, reproducible toolchain instead of whatever
happens to be installed on the host. Strictly best-effort and opt-in: it is used
only when ``environment.type`` is ``docker``/``container`` AND a usable OCI
engine is actually present. With no engine reachable (no daemon, CLI missing)
the orchestrator falls back to executing commands locally exactly as before, so
the feature can never hard-fail a build or require a daemon to exist.

Design mirrors :mod:`my_project_orchestrator.core.lsp`: detection probes are
bounded by a short timeout (a hung/absent engine can never block), every failure
is logged at debug and degrades to a safe no-op, and the runtime knob is off by
default.

Git operations (branch/commit/revert) always stay on the host; only the gate
commands are routed through the container, executed against the repository
bind-mounted at the same path so produced paths line up with the host tree.
"""

import os
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)

# Engines we know how to drive, in preference order: rootless/daemonless first
# (podman, nerdctl), then docker, then the colima-managed docker socket. All
# four speak the same ``<engine> run`` CLI surface we rely on.
_PREFERRED_ENGINES: Tuple[str, ...] = ("podman", "docker", "nerdctl", "colima")

# How long an engine availability probe may take before we treat the engine as
# unavailable. A short bound so a wedged daemon can never stall detection.
_PROBE_TIMEOUT = 8

# Default image per detected project language. Conservative, widely-mirrored
# tags; a project may override with ``environment.image``.
_LANGUAGE_IMAGES = {
    "python": "python:3.12-slim",
    "rust": "rust:slim",
    "node": "node:20-slim",
    "javascript": "node:20-slim",
    "typescript": "node:20-slim",
    "go": "golang:1.22",
    "ruby": "ruby:3.3-slim",
    "java": "eclipse-temurin:21",
}

_DEFAULT_IMAGE = "debian:stable-slim"


def _probe_engine(engine: str) -> bool:
    """True if ``engine`` is installed and its backend responds within the probe
    timeout. Never raises: any failure (missing binary, unreachable daemon,
    timeout) is reported as unavailable."""
    if engine == "colima":
        # colima is a VM manager, not a runtime CLI; a running colima exposes a
        # docker engine. Treat it as available only when its status is running.
        cmd = ["colima", "status"]
    else:
        # `info` requires a working backend (daemon for docker; none for
        # podman/nerdctl rootless), so it distinguishes "CLI present" from
        # "engine actually usable".
        cmd = [engine, "info"]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT,
        )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug(f"Container engine probe failed for {engine}: {e}")
        return False


def detect_engine(preferred: Optional[str] = None) -> Optional[str]:
    """Return the name of the first usable OCI engine, or ``None`` if none.

    ``preferred`` (from ``environment.engine``) is tried first when given; the
    standard rootless-first order follows. ``colima`` resolves to ``docker`` as
    the run CLI since colima manages a docker-compatible daemon.
    """
    order: List[str] = []
    if preferred:
        order.append(preferred)
    order.extend(e for e in _PREFERRED_ENGINES if e not in order)
    for engine in order:
        if _probe_engine(engine):
            # colima provides a docker daemon; the actual run CLI is docker.
            return "docker" if engine == "colima" else engine
    return None


def image_for_language(language: Optional[str]) -> str:
    """Pick a default base image for ``language``; falls back to a generic image
    when the language is unknown so the container can still run shell gates."""
    return _LANGUAGE_IMAGES.get((language or "").lower(), _DEFAULT_IMAGE)


class ContainerEngine:
    """Thin wrapper over a detected OCI engine for running gate commands.

    One instance per build. Commands run in a fresh ``--rm`` container with the
    repository bind-mounted at ``mount_path`` and the process mapped to the
    host uid/gid so artifacts are not left root-owned. The container is not kept
    running between commands (each gate is a self-contained ``run``), which
    keeps lifecycle trivial and leak-free while still pinning the toolchain.
    """

    def __init__(
        self,
        engine: str,
        image: str,
        host_path: Path,
        mount_path: str = "/workspace",
        network: Optional[str] = None,
        memory: Optional[str] = None,
        cpus: Optional[str] = None,
        pids_limit: Optional[int] = None,
        cap_drop: Optional[List[str]] = None,
        security_opt: Optional[List[str]] = None,
    ):
        self.engine = engine
        self.image = image
        self.host_path = Path(host_path).resolve()
        self.mount_path = mount_path
        # Egress control: "none" runs the container with no network (governance.
        # network); any other value (or None) leaves the engine default, so the
        # off path is byte-identical to before. This constrains CONTAINERIZED
        # execution only — host execution and git stay on the host network.
        self.network = network
        # Optional resource caps (environment.memory / .cpus / .pids_limit). Each
        # is emitted only when set, so an unconfigured engine produces the exact
        # same argv as before. They bound a runaway gate (fork bomb, memory hog)
        # in the throwaway container without affecting host execution.
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        # Optional sandbox hardening for running model-generated code with less
        # trust: cap_drop (e.g. ["ALL"]) drops Linux capabilities; security_opt
        # (e.g. ["no-new-privileges", "seccomp=/path/profile.json"]) passes
        # --security-opt. Both emitted only when set; a bare string is accepted
        # and wrapped, so ``cap_drop: ALL`` in YAML works too.
        self.cap_drop = [cap_drop] if isinstance(cap_drop, str) else (cap_drop or [])
        self.security_opt = (
            [security_opt] if isinstance(security_opt, str) else (security_opt or [])
        )

    def is_available(self) -> bool:
        return bool(self.engine)

    def _user_args(self) -> List[str]:
        """``--user uid:gid`` so host-bind-mounted writes are owned by the
        invoking user, not root. Skipped on platforms without uid/gid
        (Windows), where the flag is meaningless."""
        getuid = getattr(os, "getuid", None)
        getgid = getattr(os, "getgid", None)
        if getuid is None or getgid is None:
            return []
        return ["--user", f"{getuid()}:{getgid()}"]

    def wrap_command(self, command: str, timeout: int) -> List[str]:
        """Build the argv that runs ``command`` inside a throwaway container.

        The user command is passed verbatim to ``sh -c`` inside the container,
        so shell features (``&&``, pipes, globs) behave as on the host. The repo
        is the working directory via the bind mount.
        """
        argv = [
            self.engine,
            "run",
            "--rm",
            "-v",
            f"{self.host_path}:{self.mount_path}",
            "-w",
            self.mount_path,
        ]
        if self.network == "none":
            argv.extend(["--network", "none"])
        if self.memory:
            argv.extend(["--memory", str(self.memory)])
        if self.cpus:
            argv.extend(["--cpus", str(self.cpus)])
        if self.pids_limit:
            argv.extend(["--pids-limit", str(self.pids_limit)])
        for cap in self.cap_drop:
            argv.extend(["--cap-drop", str(cap)])
        for opt in self.security_opt:
            argv.extend(["--security-opt", str(opt)])
        argv.extend(self._user_args())
        argv.extend([self.image, "sh", "-c", command])
        return argv

    def run(self, command: str, timeout: int = 180) -> Tuple[bool, str]:
        """Execute ``command`` inside the container. Returns ``(ok, output)``
        with the same contract as the local runner, so callers are agnostic to
        where the command ran. Never raises: engine/timeout failures return
        ``(False, message)``."""
        argv = self.wrap_command(command, timeout)
        logger.debug(f"Container exec ({self.engine}): {shlex.join(argv)}")
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = proc.stdout
            if proc.stderr:
                output += "\n" + proc.stderr
            return proc.returncode == 0, output
        except subprocess.TimeoutExpired:
            return False, f"Container command timed out after {timeout}s: {command}"
        except (OSError, subprocess.SubprocessError) as e:
            return False, f"Container command failed: {e}"
