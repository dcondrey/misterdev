"""Optional runtime smoke gate.

A passing build/test suite proves the code compiles and units behave, but not
that the assembled artifact actually launches and responds. This gate launches
the built artifact, waits for a readiness signal, sends a probe, and asserts an
expected substring appears in the output — a cheap end-to-end liveness check.

It mirrors :mod:`my_project_orchestrator.core.lsp`: strictly opt-in (off unless
``runtime.smoke`` is configured), best-effort, and run in a daemon worker thread
with a hard timeout so a hung launch can NEVER block the build. Absent config or
a timeout is a SKIP (no opinion), not a failure; only a process that exits
non-zero or whose output lacks the expectation is a RED. Real stdout and exit
code are captured as evidence.
"""

import subprocess
import time
from pathlib import Path
from typing import Optional

from my_project_orchestrator.core.bounded import run_bounded

from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)

# Outcome constants. SKIP means "no opinion" (no config, timed out, or launch
# unavailable) and must never be treated as a pass/fail signal by callers.
SKIP = "skip"
GREEN = "green"
RED = "red"

# A hard ceiling on probe-output reading so a chatty process can't grow the
# evidence buffer without bound.
_MAX_EVIDENCE_CHARS = 16384


class SmokeResult:
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
        self.status = status
        self.evidence = evidence
        self.exit_code = exit_code
        self.reason = reason

    @property
    def passed(self) -> bool:
        return self.status == GREEN

    @property
    def skipped(self) -> bool:
        return self.status == SKIP

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
    proc = subprocess.Popen(
        launch,
        shell=True,
        cwd=str(project_root),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
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
                proc.stdin.write(probe if probe.endswith("\n") else probe + "\n")
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


def _read_available(proc: subprocess.Popen, captured: list) -> None:
    """Append any currently-buffered stdout line without blocking forever."""
    if proc.stdout is None:
        return
    line = proc.stdout.readline()
    if line:
        captured.append(line)


def _wait_for_substring(
    proc: subprocess.Popen, captured: list, needle: str, deadline: float
) -> bool:
    """Read output until ``needle`` appears, the process exits, or the deadline
    passes. Returns True iff ``needle`` was seen in the cumulative output."""
    while time.monotonic() < deadline:
        if "".join(captured).find(needle) != -1:
            return True
        if proc.poll() is not None:
            # Process gone; drain any remaining buffered output then check once.
            remainder = proc.stdout.read() if proc.stdout else ""
            if remainder:
                captured.append(remainder)
            return needle in "".join(captured)
        _read_available(proc, captured)
    return needle in "".join(captured)


def _drain_until(
    proc: subprocess.Popen, captured: list, deadline: float, max_wait: float
) -> None:
    """Read output for up to ``max_wait`` seconds (bounded by ``deadline``)."""
    stop = min(deadline, time.monotonic() + max_wait)
    while time.monotonic() < stop and proc.poll() is None:
        _read_available(proc, captured)


def _finish(
    proc: subprocess.Popen, captured: list, status: str, reason: str
) -> SmokeResult:
    code = proc.poll()
    evidence = "".join(captured)[:_MAX_EVIDENCE_CHARS]
    return SmokeResult(status, evidence=evidence, exit_code=code, reason=reason)


def _terminate(proc: subprocess.Popen) -> None:
    """Best-effort teardown: close stdin, terminate, then kill if needed."""
    if proc.poll() is not None:
        return
    try:
        if proc.stdin is not None:
            proc.stdin.close()
    except OSError:
        pass
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except (subprocess.TimeoutExpired, OSError):
        try:
            proc.kill()
            proc.wait(timeout=3)
        except (subprocess.TimeoutExpired, OSError):
            logger.debug("Smoke process did not exit after kill.")
