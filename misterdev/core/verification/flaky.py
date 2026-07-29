"""Flaky-test quarantine: don't revert a correct edit because a nondeterministic
test failed.

The G3 test gate returns RED the instant the project's test command fails. On a
repo whose suite has a flaky test (a race, a clock, a network call, an ordering
dependency) that failure is not evidence the model's edit is wrong — yet today it
reverts the edit and retries until ``max_attempts`` or the cost cap, draining
budget and, worse, discarding a correct change. This is the dominant hazard when
misterdev is pointed at a repo whose suite it does not control.

The confirmation here is a bounded, evidence-based relaxation of that gate: when
tests fail, re-run the SAME command up to ``reruns`` times with NO code change in
between. Because nothing changed, a failure that does not reproduce is
nondeterministic by definition and cannot be a deterministic consequence of the
edit — so it must not block. A failure that DOES reproduce every run is
deterministic and stays RED, exactly as before.

Correctness is preserved either way:
  - We only ever turn a RED into a GREEN when NO test failed deterministically.
  - A real regression that happens to coexist with a flake stays RED, because the
    real failure appears in every run's failing set (the intersection is
    nonempty). Precision comes from per-test identifiers; when they can't be
    parsed we degrade to a SAFE suite-level rule (any fully-passing re-run =>
    flake, otherwise stay RED), never to a wrong GREEN.

Parsing is best-effort and framework-specific (pytest, go test, cargo test,
jest). Anything unrecognized yields no identifiers and falls through to the
suite-level rule — coarser, still safe. The classification core is pure; the only
I/O is the injected ``run_fn``, mirroring the gate's ``runner`` injection so the
whole path is testable without a subprocess.
"""

import re
from dataclasses import dataclass
from typing import Callable, FrozenSet, List, Tuple

# Per-framework patterns that capture a failing test's identifier. Ordered by how
# specific/reliable the match is; every pattern is anchored on a failure keyword
# so a passing test or incidental mention is never captured. Unknown frameworks
# match nothing and degrade to the safe suite-level rule.
_FAILING_PATTERNS = (
    # pytest summary line: "FAILED tests/x.py::Cls::test_y - AssertionError"
    re.compile(r"^(?:FAILED|ERROR)\s+(\S+::\S+)", re.MULTILINE),
    # pytest verbose inline: "tests/x.py::test_y FAILED"
    re.compile(r"^(\S+::\S+)\s+(?:FAILED|ERROR)\b", re.MULTILINE),
    # go test: "--- FAIL: TestName (0.01s)"
    re.compile(r"^--- FAIL:\s+(\S+)", re.MULTILINE),
    # cargo test: "test module::test_name ... FAILED"
    re.compile(r"^test\s+(\S+)\s+\.\.\.\s+FAILED", re.MULTILINE),
    # jest: "✕ test name" / "× test name"
    re.compile(r"^\s*[✕×]\s+(.+?)\s*$", re.MULTILINE),
)


def parse_failing_tests(output: str) -> FrozenSet[str]:
    """Extract failing-test identifiers from test output, best-effort.

    Returns the set of identifiers found by the first framework pattern that
    matches anything. An empty set means "no identifiers could be parsed" — the
    caller treats that run as opaque and falls back to the suite-level rule, which
    is always safe. It never mixes patterns, so a pytest node id and a jest name
    can't be conflated into one run's set.
    """
    for pattern in _FAILING_PATTERNS:
        hits = {m.strip() for m in pattern.findall(output or "")}
        hits.discard("")
        if hits:
            return frozenset(hits)
    return frozenset()


@dataclass(frozen=True)
class RunOutcome:
    """One run of the test command: did it pass, and which tests failed.

    ``parsed`` records whether per-test identifiers were recoverable; a FAILING
    run with ``parsed=False`` is opaque (something failed, unknown what) and
    forces the safe suite-level rule.
    """

    passed: bool
    failing: FrozenSet[str] = frozenset()
    parsed: bool = False

    @staticmethod
    def of(passed: bool, output: str) -> "RunOutcome":
        if passed:
            return RunOutcome(passed=True)
        failing = parse_failing_tests(output)
        return RunOutcome(passed=False, failing=failing, parsed=bool(failing))


@dataclass(frozen=True)
class FlakyVerdict:
    """Result of confirming a red test gate.

    ``is_real_failure`` is the only value the gate acts on: True keeps RED, False
    relaxes to GREEN. ``persistent`` are the deterministically-failing tests (the
    real regressions, when identifiable); ``quarantined`` are the ones that failed
    nondeterministically and were set aside. ``reason`` is human-facing.
    """

    is_real_failure: bool
    persistent: FrozenSet[str] = frozenset()
    quarantined: FrozenSet[str] = frozenset()
    reason: str = ""


def classify(outcomes: List[RunOutcome]) -> FlakyVerdict:
    """Pure classification of repeated runs of the SAME test command.

    Preconditions: ``outcomes`` is the ordered list of every run (the original red
    run first, then the re-runs); it holds at least one run and the first run
    failed. No code changes between runs, so any variation is nondeterminism.
    """
    if not outcomes:
        # No evidence at all: caller had a red run, so default to keeping RED.
        return FlakyVerdict(is_real_failure=True, reason="no runs to classify")

    failing_runs = [o for o in outcomes if not o.passed]
    if not failing_runs:
        # Every run passed on re-run: the original failure never reproduced.
        return FlakyVerdict(
            is_real_failure=False,
            reason=f"test failure did not reproduce across {len(outcomes)} runs",
        )

    if all(o.parsed for o in failing_runs):
        # Per-test precision: a test failing in EVERY run is deterministic. Passing
        # runs contribute an empty failing set, so a test that passed on any re-run
        # drops out of the intersection — exactly the flake we want to quarantine.
        persistent = frozenset.intersection(*(o.failing for o in outcomes))
        union = frozenset.union(*(o.failing for o in outcomes))
        quarantined = union - persistent
        if persistent:
            return FlakyVerdict(
                is_real_failure=True,
                persistent=persistent,
                quarantined=quarantined,
                reason=(
                    f"{len(persistent)} test(s) failed deterministically across "
                    f"{len(outcomes)} runs"
                    + (f"; quarantined {len(quarantined)} flaky" if quarantined else "")
                ),
            )
        any_passed = any(o.passed for o in outcomes)
        return FlakyVerdict(
            is_real_failure=not any_passed,
            quarantined=quarantined,
            reason=(
                f"no test failed in every run; {len(quarantined)} failed "
                f"nondeterministically across {len(outcomes)} runs"
            ),
        )

    # At least one failing run was opaque (couldn't identify its tests). We cannot
    # trust an id-intersection, so fall back to the suite-level rule: if any run
    # passed cleanly the failure is nondeterministic; otherwise keep RED.
    any_passed = any(o.passed for o in outcomes)
    return FlakyVerdict(
        is_real_failure=not any_passed,
        reason=(
            "test failure did not reproduce (a clean re-run passed)"
            if any_passed
            else f"tests failed in every one of {len(outcomes)} runs"
        ),
    )


def confirm_test_failure(
    run_fn: Callable[[], Tuple[bool, str]],
    first_output: str,
    reruns: int,
) -> FlakyVerdict:
    """Confirm a red test gate by re-running the command up to ``reruns`` times.

    ``run_fn`` re-executes the SAME test command and returns ``(passed, output)``
    — the gate injects a closure over ``_run_cmd`` so this stays testable without a
    subprocess. Re-running stops early the moment a deterministic verdict is
    certain: if a re-run passes cleanly the failure is already proven
    nondeterministic. ``reruns <= 0`` disables confirmation and keeps the original
    RED (the caller normally guards on this too).
    """
    outcomes: List[RunOutcome] = [RunOutcome.of(False, first_output)]
    if reruns <= 0:
        return FlakyVerdict(is_real_failure=True, reason="flaky confirmation disabled")

    for _ in range(reruns):
        passed, output = run_fn()
        outcomes.append(RunOutcome.of(passed, output))
        if passed:
            # A clean re-run is decisive: the original failure did not reproduce.
            break

    return classify(outcomes)
