"""Optional runtime smoke gate.

A passing build/test suite proves the code compiles and units behave, but not
that the assembled artifact actually launches and responds. This gate launches
the built artifact, waits for a readiness signal, sends a probe, and asserts an
expected substring appears in the output — a cheap end-to-end liveness check.

It mirrors :mod:`misterdev.core.context.lsp`: strictly opt-in (off unless
``runtime.smoke`` is configured), best-effort, and run in a daemon worker thread
with a hard timeout so a hung launch can NEVER block the build. Absent config or
a timeout is a SKIP (no opinion), not a failure; only a process that exits
non-zero or whose output lacks the expectation is a RED. Real stdout and exit
code are captured as evidence.
"""

import os
import select
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

from misterdev.core.execution.bounded import run_bounded
from misterdev.core.execution.outcomes import (
    GREEN,
    RED,
    SKIP,
    GateOutcome,
)

from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)

# Outcome constants. SKIP means "no opinion" (no config, timed out, or launch
# unavailable) and must never be treated as a pass/fail signal by callers.
# A hard ceiling on probe-output reading so a chatty process can't grow the
# evidence buffer without bound.
_MAX_EVIDENCE_CHARS = 16384


class SmokeResult(GateOutcome):
    """Outcome of a smoke run. ``status`` is SKIP/GREEN/RED; ``evidence`` is the
    captured stdout/stderr (truncated); ``exit_code`` is the process exit code
    when it terminated, else ``None``."""

    def __init__(
        self,
        status: str,
        evidence: str = "",
        exit_code: Optional[int] = None,
        reason: str = "",
    ):
        super().__init__(status, reason)
        self.evidence = evidence
        self.exit_code = exit_code

    def __repr__(self) -> str:
        return f"SmokeResult(status={self.status!r}, exit_code={self.exit_code!r})"


def run_smoke_gate(
    project_root: Path,
    smoke_config: Optional[dict],
    runner=None,
) -> SmokeResult:
    """Run the smoke gate described by ``smoke_config``.

    ``smoke_config`` keys:
      - ``launch`` (required): command that starts the artifact.
      - ``ready`` (optional): substring that, once seen in output, means ready.
      - ``probe`` (optional): text piped to the process's stdin after readiness.
      - ``expect`` (required for a verdict): substring asserted in the output.
      - ``timeout`` (optional, default 30): hard ceiling for the whole run.

    Returns a :class:`SmokeResult`. SKIP when there is no config or no
    ``launch`` (feature off), or when the hard timeout fires (never blocks).
    ``runner`` is accepted for symmetry with the gate seam but smoke launches
    run on the host (where the artifact and its ports live); it is unused today.
    """
    if not smoke_config or not smoke_config.get("launch"):
        return SmokeResult(SKIP, reason="no runtime.smoke config")

    launch = smoke_config["launch"]
    ready = smoke_config.get("ready")
    probe = smoke_config.get("probe")
    expect = smoke_config.get("expect")
    timeout = float(smoke_config.get("timeout", 30))

    def _work() -> SmokeResult:
        try:
            return _smoke(project_root, launch, ready, probe, expect, timeout)
        except Exception as e:  # any launch/IO failure is non-fatal -> skip
            logger.debug(f"Runtime smoke gate unavailable: {e}")
            return SmokeResult(SKIP, reason=f"error: {e}")

    # A small margin over the inner timeout so a clean inner teardown is
    # preferred, but the outer bound still guarantees we return.
    return run_bounded(
        _work, timeout + 5, SmokeResult(SKIP, reason="timed out"), "Runtime smoke gate"
    )


def _smoke(
    project_root: Path,
    launch: str,
    ready: Optional[str],
    probe: Optional[str],
    expect: Optional[str],
    timeout: float,
) -> SmokeResult:
    """Launch, await readiness, probe, assert. Always tears the process down."""
    deadline = time.monotonic() + timeout
    # Binary, unbuffered pipes so reads go through select()+os.read and are
    # deadline-bounded (a text-mode readline() blocks until a newline, defeating
    # the inner timeout). start_new_session puts the launch in its own process
    # group so teardown can kill the whole tree, not just the shell.
    proc = subprocess.Popen(
        launch,
        shell=True,
        cwd=str(project_root),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    captured: list[str] = []
    try:
        if ready:
            if not _wait_for_substring(proc, captured, ready, deadline):
                return _finish(proc, captured, RED, "readiness signal not seen")
        elif probe is None:
            # No readiness signal and no probe: give the process a brief moment
            # to emit startup output before we evaluate it.
            _drain_until(proc, captured, deadline, max_wait=min(2.0, timeout))

        if probe is not None and proc.stdin is not None:
            try:
                data = probe if probe.endswith("\n") else probe + "\n"
                proc.stdin.write(data.encode("utf-8"))
                proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass

        if expect:
            seen = _wait_for_substring(proc, captured, expect, deadline)
            status = GREEN if seen else RED
            reason = "" if seen else f"expected substring {expect!r} not found"
            return _finish(proc, captured, status, reason)

        # No expectation to assert: GREEN if it launched and is still healthy
        # (or exited cleanly).
        _drain_until(proc, captured, deadline, max_wait=0.2)
        code = proc.poll()
        if code is not None and code != 0:
            return _finish(proc, captured, RED, f"exited {code}")
        return _finish(proc, captured, GREEN, "")
    finally:
        _terminate(proc)


def _read_available(proc: subprocess.Popen, captured: list, deadline: float) -> None:
    """Append whatever stdout is ready, waiting at most until ``deadline``.

    Uses ``select`` + ``os.read`` so a process that emits no newline (or nothing
    at all) can never block past the deadline — unlike ``readline()``, which the
    inner timeout was silently relying on and which made the timeout a no-op.
    """
    fd = proc.stdout
    if fd is None:
        return
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return
    try:
        ready, _, _ = select.select([fd], [], [], min(remaining, 0.1))
    except (OSError, ValueError):
        return
    if not ready:
        return
    try:
        chunk = os.read(fd.fileno(), 65536)
    except (OSError, ValueError):
        return
    if chunk:
        captured.append(chunk.decode("utf-8", "replace"))


def _drain_remaining(proc: subprocess.Popen, captured: list) -> None:
    """Read all output still buffered after the process has exited (non-blocking)."""
    fd = proc.stdout
    if fd is None:
        return
    while True:
        try:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                return
            chunk = os.read(fd.fileno(), 65536)
        except (OSError, ValueError):
            return
        if not chunk:  # EOF
            return
        captured.append(chunk.decode("utf-8", "replace"))


def _wait_for_substring(
    proc: subprocess.Popen, captured: list, needle: str, deadline: float
) -> bool:
    """Read output until ``needle`` appears, the process exits, or the deadline
    passes. Returns True iff ``needle`` was seen in the cumulative output."""
    buf = "".join(captured)
    joined_up_to = len(captured)
    if needle in buf:
        return True
    while time.monotonic() < deadline:
        _read_available(proc, captured, deadline)
        if len(captured) > joined_up_to:
            buf += "".join(captured[joined_up_to:])
            joined_up_to = len(captured)
        if needle in buf:
            return True
        if proc.poll() is not None:
            # Process gone; drain any remaining buffered output then check once.
            _drain_remaining(proc, captured)
            if len(captured) > joined_up_to:
                buf += "".join(captured[joined_up_to:])
            return needle in buf
    if len(captured) > joined_up_to:
        buf += "".join(captured[joined_up_to:])
    return needle in buf


def _drain_until(
    proc: subprocess.Popen, captured: list, deadline: float, max_wait: float
) -> None:
    """Read output for up to ``max_wait`` seconds (bounded by ``deadline``)."""
    stop = min(deadline, time.monotonic() + max_wait)
    while time.monotonic() < stop and proc.poll() is None:
        _read_available(proc, captured, stop)
    if proc.poll() is not None:
        _drain_remaining(proc, captured)


def _finish(
    proc: subprocess.Popen, captured: list, status: str, reason: str
) -> SmokeResult:
    code = proc.poll()
    evidence = "".join(captured)[:_MAX_EVIDENCE_CHARS]
    return SmokeResult(status, evidence=evidence, exit_code=code, reason=reason)


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    """Send ``sig`` to the launch's whole process group so a server that forked
    its own children (e.g. ``npm start`` -> node worker) is torn down too, not
    just the shell. Falls back to signalling the direct child where process
    groups are unavailable (Windows) or already gone."""
    if hasattr(os, "killpg"):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        proc.terminate() if sig == signal.SIGTERM else proc.kill()
    except OSError:
        pass


def _terminate(proc: subprocess.Popen) -> None:
    """Best-effort teardown: close stdin, SIGTERM the group, then SIGKILL it."""
    if proc.poll() is not None:
        return
    try:
        if proc.stdin is not None:
            proc.stdin.close()
    except OSError:
        pass
    _signal_group(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=3)
        return
    except (subprocess.TimeoutExpired, OSError):
        pass
    _signal_group(proc, signal.SIGKILL)
    try:
        proc.wait(timeout=3)
    except (subprocess.TimeoutExpired, OSError):
        logger.debug("Smoke process group did not exit after kill.")
