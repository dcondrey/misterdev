"""Process-signal early-abort: stop a task that is not converging.

Today a task retries until an outcome gate fails ``max_attempts`` times or the
cost cap trips — even when the PROCESS signal already says it is stuck: the same
assertion failing attempt after attempt, or the diagnostic count refusing to
shrink. Those attempts are near-certain to fail and burn budget that a different
task (or a stronger tier, or a human) could use (docs/research-directions.md,
Theme 5 — PRM course-correction 2509.02360).

This is a pure, deterministic monitor over per-attempt signals. It NEVER changes
correctness (a task it abandons was already failing every gate); it only decides,
early, to stop pouring attempts into a non-converging task. Two independent
triggers:

  - **stuck**: the same normalized error fingerprint recurs ``stuck_repeats``
    times in a row — the model is circling, not fixing.
  - **no progress**: the diagnostic/failure count fails to strictly decrease
    across a window of ``no_progress_window`` attempts — no forward motion.

Fingerprints are normalized (paths, line numbers, hex addresses, and quoted
literals stripped) so "the same error" is recognized across attempts whose
incidental detail differs. Pure and side-effect free.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

_NORMALIZERS = (
    (re.compile(r"0x[0-9a-fA-F]+"), "0xHEX"),
    (re.compile(r"\b\d+\b"), "N"),  # line numbers, counts, offsets
    (re.compile(r'"[^"]*"|\'[^\']*\''), "STR"),  # quoted literals
    (re.compile(r"(/[^\s:]+)+"), "PATH"),  # file paths
    (re.compile(r"\s+"), " "),
)


def fingerprint(error_text: str) -> str:
    """A stable fingerprint of an error, ignoring incidental detail so the SAME
    failure is recognized across attempts."""
    text = (error_text or "").strip().lower()
    for pattern, repl in _NORMALIZERS:
        text = pattern.sub(repl, text)
    return text.strip()[:400]


@dataclass
class AttemptSignal:
    """One attempt's process signal: how many diagnostics, and their fingerprint."""

    error_count: int
    error_text: str = ""

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.error_text)


@dataclass
class ConvergenceMonitor:
    """Accumulates per-attempt signals and reports when to abort early.

    ``stuck_repeats`` and ``no_progress_window`` count ATTEMPTS, not calls; both
    require at least that many signals before they can fire, so a monitor never
    aborts before it has evidence. Disabled triggers: set either to 0.
    """

    stuck_repeats: int = 3
    no_progress_window: int = 3
    _signals: List[AttemptSignal] = field(default_factory=list)

    def update(self, signal: AttemptSignal) -> None:
        self._signals.append(signal)

    def _stuck(self) -> bool:
        if self.stuck_repeats < 2 or len(self._signals) < self.stuck_repeats:
            return False
        recent = self._signals[-self.stuck_repeats :]
        first = recent[0].fingerprint
        return bool(first) and all(s.fingerprint == first for s in recent)

    def _no_progress(self) -> bool:
        if self.no_progress_window < 2 or len(self._signals) < self.no_progress_window:
            return False
        counts = [s.error_count for s in self._signals[-self.no_progress_window :]]
        # No strict decrease anywhere in the window -> not making forward motion.
        return all(b >= a for a, b in zip(counts, counts[1:]))

    def should_abort(self) -> Tuple[bool, Optional[str]]:
        """``(abort, reason)`` — advisory; the caller decides what to do (escalate
        tier, backtrack, or stop). A task that has never had a signal never aborts."""
        if self._stuck():
            fp = self._signals[-1].fingerprint
            return True, f"stuck: same error {self.stuck_repeats}x in a row ({fp[:80]})"
        if self._no_progress():
            counts = [s.error_count for s in self._signals[-self.no_progress_window :]]
            return True, f"no progress: diagnostics not shrinking over {counts}"
        return False, None
