"""Optional mutation-score gate (generic, command-based, language-agnostic).

A passing test suite proves the tests *run* green, but not that they would
*catch a regression*: a suite with no meaningful assertions passes while a
mutation testing tool shows it kills almost no injected faults. This gate runs a
project-configured mutation command, parses a mutation score from its output,
and asserts it meets a floor.

It is deliberately tool-agnostic: it does NOT hardcode mutmut / cosmic-ray /
Stryker / cargo-mutants. The project supplies the command and the floor; this
module just runs it and robustly parses a percentage or fraction from common
output formats, skipping cleanly when the score is unparseable.

It mirrors :mod:`my_project_orchestrator.core.execution.runtime`: strictly opt-in (off
unless ``mutation.command`` is configured and ``orchestrator.mutation_gate`` is
true), best-effort, and run in a daemon worker thread with a hard timeout so a
slow or hung mutation run can NEVER block the build. Absent config or a timeout
is a SKIP (no opinion); an unparseable score is ALSO a SKIP (we will not fail a
build on a number we could not read); only a parsed score strictly below the
floor is a RED.
"""

import re
import subprocess
from pathlib import Path
from typing import Optional

from my_project_orchestrator.core.execution.bounded import run_bounded
from my_project_orchestrator.core.execution.outcomes import (
    GREEN,
    RED,
    SKIP,
    GateOutcome,
)
from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)

# Outcome constants. SKIP means "no opinion" (no config, unparseable score, or
# timeout) and must never be treated as a pass/fail signal by callers.
_MAX_EVIDENCE_CHARS = 16384

# Score patterns, tried in order. Each captures a number normalized to a 0..1
# fraction by :func:`_parse_score`. Ordered most-specific first so a labeled
# score ("mutation score: 82%") wins over a bare trailing percentage.
_SCORE_PATTERNS = (
    # "mutation score: 82.5%" / "score = 82%" / "killed 82.5 %"
    re.compile(
        r"(?:mutation\s+)?score[^0-9]{0,12}([0-9]+(?:\.[0-9]+)?)\s*%",
        re.IGNORECASE,
    ),
    # "mutation score: 0.82" / "score = .82" (fraction, no percent sign)
    re.compile(
        r"(?:mutation\s+)?score[^0-9]{0,12}([01](?:\.[0-9]+)?|\.[0-9]+)\b",
        re.IGNORECASE,
    ),
    # "killed 41/50" / "41 / 50 mutants killed" (a ratio)
    re.compile(r"([0-9]+)\s*/\s*([0-9]+)"),
    # Fallback: a standalone percentage anywhere ("82.5%").
    re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%"),
)


class MutationResult(GateOutcome):
    """Outcome of a mutation-score gate run.

    ``status`` is SKIP/GREEN/RED; ``score`` is the parsed mutation score as a
    0..1 fraction (None when unparseable/skipped); ``evidence`` is the captured
    output (truncated); ``reason`` explains a SKIP/RED.
    """

    def __init__(
        self,
        status: str,
        score: Optional[float] = None,
        evidence: str = "",
        reason: str = "",
    ):
        super().__init__(status, reason)
        self.score = score
        self.evidence = evidence

    def __repr__(self) -> str:
        return f"MutationResult(status={self.status!r}, score={self.score!r})"


def run_mutation_gate(
    project_root: Path,
    mutation_config: Optional[dict],
    runner=None,
) -> MutationResult:
    """Run the mutation gate described by ``mutation_config``.

    ``mutation_config`` keys:
      - ``command`` (required): the mutation-testing command to run. No command
        means the feature is off -> SKIP.
      - ``min_score`` (optional, default 0.0): the floor the parsed score must
        meet, accepted as a fraction (0.8) or a percentage (80). A score at or
        above the floor is GREEN, strictly below is RED.
      - ``timeout`` (optional, default 1800): hard ceiling for the whole run, in
        seconds (mutation runs are slow). The outer join always returns.

    Returns a :class:`MutationResult`. ``runner`` accepts the gate's container
    seam ``(cmd, timeout) -> (ok, output)``; when None the command runs locally.
    """
    if not mutation_config or not mutation_config.get("command"):
        return MutationResult(SKIP, reason="no mutation config")

    command = mutation_config["command"]
    min_score = _normalize_floor(mutation_config.get("min_score", 0.0))
    timeout = float(mutation_config.get("timeout", 1800))

    def _work() -> MutationResult:
        try:
            return _mutation(project_root, command, min_score, timeout, runner)
        except Exception as e:  # any launch/IO failure is non-fatal -> skip
            logger.debug(f"Mutation gate unavailable: {e}")
            return MutationResult(SKIP, reason=f"error: {e}")

    # A small margin over the inner timeout so a clean inner teardown is
    # preferred, but the outer bound still guarantees we return.
    return run_bounded(
        _work, timeout + 5, MutationResult(SKIP, reason="timed out"), "Mutation gate"
    )


def _mutation(
    project_root: Path,
    command: str,
    min_score: float,
    timeout: float,
    runner,
) -> MutationResult:
    """Run the command, parse a score, and compare it to the floor.

    A non-zero exit is not itself a failure: many mutation tools exit non-zero
    when survivors exist yet still print a usable score, so we parse the output
    regardless and only RED on a score below the floor. We SKIP (not RED) when no
    score can be parsed, so the build is never failed on an unread number.
    """
    if runner is not None:
        ok, output = runner(command, int(timeout))
    else:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return MutationResult(SKIP, reason="command timed out")
        output = (proc.stdout or "") + (proc.stderr or "")

    evidence = output[-_MAX_EVIDENCE_CHARS:]
    score = parse_mutation_score(output)
    if score is None:
        return MutationResult(
            SKIP, evidence=evidence, reason="could not parse a mutation score"
        )
    if score >= min_score:
        return MutationResult(GREEN, score=score, evidence=evidence)
    return MutationResult(
        RED,
        score=score,
        evidence=evidence,
        reason=f"mutation score {score:.2%} below floor {min_score:.2%}",
    )


def parse_mutation_score(output: str) -> Optional[float]:
    """Parse a mutation score from ``output`` as a 0..1 fraction, or None.

    Handles labeled percentages ("mutation score: 82.5%"), labeled fractions
    ("score: 0.82"), kill ratios ("killed 41/50"), and a bare trailing
    percentage. Returns None when nothing parseable is found OR the value is out
    of the sane 0..1 range, so a stray number can't be misread as a score.
    """
    if not output:
        return None
    for pattern in _SCORE_PATTERNS:
        match = pattern.search(output)
        if not match:
            continue
        groups = match.groups()
        if len(groups) == 2:  # ratio "killed/total"
            killed = float(groups[0])
            total = float(groups[1])
            if total <= 0:
                continue
            value = killed / total
        else:
            raw = float(groups[0])
            # A value > 1 is a percentage even without a sign (the percent
            # patterns already matched the sign); normalize to a fraction.
            value = raw / 100.0 if raw > 1 else raw
        if 0.0 <= value <= 1.0:
            return value
    return None


def _normalize_floor(value) -> float:
    """Accept a floor as a fraction (0.8) or a percentage (80) -> 0..1 fraction.

    A value > 1 is read as a percentage. Out-of-range or non-numeric input
    clamps to a 0.0 floor (effectively "any score passes") rather than raising,
    so a misconfigured floor degrades to permissive instead of crashing.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if f > 1.0:
        f = f / 100.0
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f
