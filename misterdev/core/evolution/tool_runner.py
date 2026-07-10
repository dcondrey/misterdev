"""Sandboxed execution of model-authored (UNTRUSTED) tools — the runtime half of
two-timescale evolution (see ``docs/two-timescale-evolution.md``).

live-SWE-agent's edge is that the agent writes its own task-specific helper tools
at runtime. Those tools are MODEL-AUTHORED CODE and are therefore untrusted. This
module runs them at the strictest reuse of the existing
:class:`~misterdev.core.execution.container.ContainerEngine`, which was built for
exactly this ("running model-generated code with less trust"): no network, all
Linux capabilities dropped, ``no-new-privileges``, memory/CPU/PID caps, a
wall-clock timeout, an isolated working directory that is NOT the project tree,
and ``--rm`` cleanup.

Security invariant — non-negotiable:

* Model-authored tool code NEVER executes on the host and NEVER sees the project
  repository. It runs only inside the hardened, network-less container over a
  throwaway temp dir.
* With no container engine available the tool is NOT run (status ``skip``). There
  is no host fallback: an unsandboxed host exec of untrusted code is never
  acceptable, so the capability simply degrades off.
* A tool is a pure function: the model supplies its stdin, the sandbox returns
  stdout/stderr/exit. The tool cannot roam the filesystem or reach the network to
  read code or exfiltrate.

Deterministic and offline-testable: the container invocation is injected, so the
size/limit/skip/classification logic is unit-tested without Docker; the real
factory is a thin adapter over ``ContainerEngine``.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol, Tuple

from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)

# Bound the untrusted payload and its captured output. A tool is a small helper,
# not a program; these caps keep a giant source or a runaway printer from
# ballooning memory/context even inside the sandbox.
_MAX_SOURCE_BYTES = 64 * 1024
_MAX_OUTPUT_CHARS = 16 * 1024
_DEFAULT_IMAGE = "python:3.12-slim"


@dataclass(frozen=True)
class ToolRunResult:
    """Outcome of running one model-authored tool in the sandbox."""

    status: str  # "ok" | "error" | "timeout" | "skip" | "rejected"
    stdout: str
    stderr: str
    exit_code: Optional[int] = None

    @property
    def ran(self) -> bool:
        """True when the tool actually executed (regardless of exit status)."""
        return self.status in ("ok", "error", "timeout")

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class _EngineLike(Protocol):
    def run(self, command: str, timeout: int) -> Tuple[bool, str]: ...


# engine_factory(work_dir) -> an engine bound to that dir, or None when no
# container sandbox is available. Injected so tests never touch Docker.
EngineFactory = Callable[[Path], Optional[_EngineLike]]


def _default_engine_factory(
    image: str,
    memory: str,
    cpus: str,
    pids_limit: int,
) -> EngineFactory:
    """Build a factory that returns a MAXIMALLY-hardened ContainerEngine over a
    throwaway dir, or None when no engine is available (capability degrades off).
    """

    def factory(work_dir: Path) -> Optional[_EngineLike]:
        from misterdev.core.execution.container import ContainerEngine, detect_engine

        engine = detect_engine()
        if engine is None:
            return None
        return ContainerEngine(
            engine=engine,
            image=image,
            host_path=work_dir,
            mount_path="/work",
            network="none",  # no egress: cannot exfiltrate or fetch
            memory=memory,
            cpus=cpus,
            pids_limit=pids_limit,
            cap_drop=["ALL"],  # drop every Linux capability
            security_opt=["no-new-privileges"],
        )

    return factory


class ToolRunner:
    """Runs untrusted, model-authored Python tools in a hardened sandbox."""

    def __init__(
        self,
        engine_factory: Optional[EngineFactory] = None,
        *,
        image: str = _DEFAULT_IMAGE,
        timeout: int = 20,
        memory: str = "256m",
        cpus: str = "1",
        pids_limit: int = 128,
    ):
        self.timeout = timeout
        self._factory = engine_factory or _default_engine_factory(
            image, memory, cpus, pids_limit
        )

    def run(self, source: str, stdin: str = "") -> ToolRunResult:
        """Execute ``source`` as a Python tool in the sandbox with ``stdin`` piped
        in. Never raises; every failure mode is a status, not an exception.

        Returns ``skip`` when no sandbox is available (the code is never run on the
        host), ``rejected`` when the source is empty or over the size cap, and
        otherwise ``ok``/``error``/``timeout`` from the sandboxed run.
        """
        if not isinstance(source, str) or not source.strip():
            return ToolRunResult("rejected", "", "empty tool source")
        if len(source.encode("utf-8", "ignore")) > _MAX_SOURCE_BYTES:
            return ToolRunResult(
                "rejected", "", f"tool source exceeds {_MAX_SOURCE_BYTES}-byte cap"
            )
        try:
            with tempfile.TemporaryDirectory(prefix="mdev-tool-") as td:
                work = Path(td)
                (work / "tool.py").write_text(source, encoding="utf-8")
                (work / "stdin").write_text(stdin or "", encoding="utf-8")
                engine = self._factory(work)
                if engine is None:
                    return ToolRunResult(
                        "skip", "", "no container sandbox available; tool not run"
                    )
                # Redirect the model-supplied stdin from the mounted file. The
                # command runs as the container's `sh -c`, cwd = the mount.
                ok, output = engine.run(
                    "python /work/tool.py < /work/stdin", timeout=self.timeout
                )
        except OSError as e:  # temp-dir/write failure — never propagate
            logger.debug(f"Tool sandbox setup failed: {e}")
            return ToolRunResult("skip", "", f"sandbox setup failed: {e}")

        output = output or ""
        if len(output) > _MAX_OUTPUT_CHARS:
            head = output[: _MAX_OUTPUT_CHARS // 2]
            tail = output[-_MAX_OUTPUT_CHARS // 2 :]
            output = f"{head}\n...[{len(output) - _MAX_OUTPUT_CHARS} chars elided]...\n{tail}"
        if ok:
            return ToolRunResult("ok", output, "", exit_code=0)
        if "timed out" in output.lower():
            return ToolRunResult("timeout", "", output, exit_code=None)
        return ToolRunResult("error", output, output, exit_code=1)
