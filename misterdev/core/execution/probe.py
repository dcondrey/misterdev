"""Failure-triggered single-test re-run to capture fresh, focused ground truth.

``failure_view`` parses a *whole* gate's output into per-test records, but that
output is stale by the time the model sees it: mixed with every other test, and
already compressed. When a gate goes red the most useful signal is the ONE
failing test re-run in isolation — a clean, uncluttered trace of exactly what
still breaks.

This is deliberately NOT the old always-on probe system that re-ran probes on
every step and blew up the token budget. This one fires ONCE, only on a real
gate failure, on a SINGLE test: bounded cost, additive signal. If the runner is
unrecognized or the re-run errors out, the caller falls back to the existing
compressed error context, so this can only add signal, never remove the fallback.

Pure command construction (``isolate_command``) is the tested core; the actual
subprocess (``run_probe``) mirrors ``misterdev.tools.command`` — own process
group, SIGKILL the whole tree on timeout, and never raise.
"""

import shlex
import subprocess
from typing import Optional

from misterdev.logging_setup import setup_logger
from misterdev.utils.process import kill_process_group

logger = setup_logger(__name__)


def _quote(name: str) -> str:
    """Escape a test name for a shell double-quoted argument (jest/dotnet)."""
    # Backslash-escape the characters that stay special inside "..." so a test
    # name containing them can't break out of the quoting or inject a command.
    return (
        name.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("$", "\\$")
        .replace("`", "\\`")
    )


def isolate_command(
    base_test_command: str, test_name: str, language: str
) -> Optional[str]:
    """Build a command that runs ONLY ``test_name`` from the project's runner.

    ``base_test_command`` is the full gate command (e.g. ``"python -m pytest -q"``,
    ``"cargo test"``, ``"npm test"``, ``"swift test"``, ``"dotnet test"``); the
    runner is inferred from ``language`` and the command text. Returns the
    isolated command, or ``None`` when the runner is unrecognized (the caller
    then skips the probe and keeps the compressed view).

    Per-runner isolate flags:
      pytest  -> ``<base> -k <name>``
      cargo   -> ``cargo test <name>``
      jest/vitest (npm) -> ``npm test -- -t "<name>"``
      swift   -> ``swift test --filter <name>``
      dotnet  -> ``dotnet test --filter "FullyQualifiedName~<name>"``
    """
    base = (base_test_command or "").strip()
    name = (test_name or "").strip()
    if not base or not name:
        return None

    lang = (language or "").lower()
    lowered = base.lower()

    # Detect by explicit language first, then by the command text, so an
    # ambiguous/empty language still resolves off a recognizable runner.
    if lang == "python" or "pytest" in lowered:
        # -k takes a substring/expression; a shell-safe single token suffices.
        return f"{base} -k {shlex.quote(name)}"

    if lang == "rust" or "cargo" in lowered:
        # `cargo test <name>` filters by substring; drop base flags to avoid
        # re-declaring the whole workspace's tests.
        return f"cargo test {shlex.quote(name)}"

    if (
        lang in ("javascript", "typescript")
        or "npm" in lowered
        or "jest" in lowered
        or "vitest" in lowered
    ):
        # `--` forwards `-t` to the underlying jest/vitest runner, not to npm.
        return f'npm test -- -t "{_quote(name)}"'

    if lang == "swift" or "swift" in lowered:
        return f"swift test --filter {shlex.quote(name)}"

    if lang == "csharp" or "dotnet" in lowered:
        return f'dotnet test --filter "FullyQualifiedName~{_quote(name)}"'

    return None


def run_probe(cwd: str, command: str, timeout: int = 120) -> Optional[str]:
    """Run ``command`` in ``cwd``, bounded by ``timeout``; return combined
    stdout+stderr, or ``None`` on any error or timeout. Never raises.

    Mirrors ``misterdev.tools.command``: ``start_new_session`` puts the run in
    its own process group so a timeout SIGKILLs the whole tree (test workers,
    servers) rather than orphaning grandchildren; ``errors="replace"`` keeps a
    stray non-UTF8 byte from raising.
    """
    if not command:
        return None
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        )
    except Exception as e:  # spawn failure (bad cwd, exec error): degrade quietly
        logger.warning(f"Probe failed to start '{command}': {e}")
        return None

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_process_group(proc)
        # Reap the killed group, but bound the wait so a double-forked
        # grandchild holding the pipe open can't hang the probe indefinitely.
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
        logger.warning(f"Probe timed out after {timeout}s: {command}")
        return None
    except Exception as e:  # any other communicate failure: never propagate
        logger.warning(f"Probe errored '{command}': {e}")
        return None

    output = stdout or ""
    if stderr:
        output += "\n" + stderr
    return output
