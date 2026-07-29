"""Structured extraction of the *exact* failing assertions from a test run.

An agent can only fix what it can see. The prior failure path compressed and
truncated raw runner output (and the classifier could mislabel a test failure
as "syntax"), so on a red gate the model got a lossy view of ground truth — the
observed cause of stuck exact-string/edge-case failures. This seam parses a
runner's output into ``Failure`` records (test name, location, expected vs
actual) and renders a tight, exact block that leads the retry context.

Deterministic and offline-testable: parsers are validated against captured real
pytest/jest/cargo-test output, not assumed formats. Unknown output yields no
failures (the caller falls back to the existing compressed view), so this can
only add signal, never remove the fallback.
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Failure:
    """One failing test, reduced to what the model needs to fix it."""

    test: str
    expected: Optional[str] = None
    actual: Optional[str] = None
    message: Optional[str] = None
    location: Optional[str] = None  # "file:line"


# --- pytest -----------------------------------------------------------------

# `FAILED path::Class::test - AssertionError: <msg>` (the -q summary tail).
_PYTEST_SUMMARY = re.compile(
    r"^FAILED\s+(?P<path>\S+?)::(?P<test>[\w.\[\]-]+)\s*(?:-\s*(?P<msg>.*))?$"
)
# A FAILURES-section header: `____ Class.test_name ____`.
_PYTEST_HEADER = re.compile(r"^_{4,}\s+(?P<test>\S+)\s+_{4,}$")
# The `E   AssertionError: <actual> != <expected>` assertion line (richer than
# the truncated summary tail). unittest assertEqual renders `<actual> != <expected>`.
_PYTEST_E = re.compile(r"^E\s+(?P<exc>\w*Error): (?P<detail>.+)$")
_PYTEST_NE = re.compile(r"^(?P<actual>.+?) != (?P<expected>.+)$")


def _parse_pytest(output: str) -> List[Failure]:
    lines = output.splitlines()
    out: List[Failure] = []
    seen = set()
    # Prefer the FAILURES section: each `__ test __` header paired with its first
    # `E  <Exc>: ...` line carries pytest's head/tail-truncated actual != expected.
    for i, line in enumerate(lines):
        h = _PYTEST_HEADER.match(line)
        if not h:
            continue
        test = h.group("test").split(".")[-1]
        f = Failure(test=test)
        for w in lines[i + 1 :]:  # scan to the E line; break at the next header
            e = _PYTEST_E.match(w)
            if e:
                detail = e.group("detail").strip()
                ne = _PYTEST_NE.match(detail)
                if ne:
                    f.actual = ne.group("actual").strip()
                    f.expected = ne.group("expected").strip()
                else:
                    f.message = f"{e.group('exc')}: {detail}"
                break
            if _PYTEST_HEADER.match(w):
                break
        out.append(f)
        seen.add(test)
    # Fallback: the `FAILED path::test - msg` summary tail (present under -q even
    # when no FAILURES bodies were emitted).
    for line in lines:
        m = _PYTEST_SUMMARY.match(line.strip())
        if not m or m.group("test").split(".")[-1] in seen:
            continue
        msg = (m.group("msg") or "").strip()
        f = Failure(test=m.group("test").split(".")[-1], message=msg or None)
        ne = _PYTEST_NE.match(re.sub(r"^\w*Error:?\s*", "", msg))
        if ne:
            f.actual = ne.group("actual").strip()
            f.expected = ne.group("expected").strip()
        out.append(f)
    return out


# --- jest -------------------------------------------------------------------

_JEST_HEADER = re.compile(r"^\s*[●✕]\s+(?P<test>.+?)\s*$")
_JEST_EXPECTED = re.compile(r"^\s*Expected:\s*(?P<v>.+?)\s*$")
_JEST_RECEIVED = re.compile(r"^\s*Received:\s*(?P<v>.+?)\s*$")
_JEST_AT = re.compile(r"\bat\s+.*\((?P<loc>[^()]+:\d+:\d+)\)")


def _parse_jest(output: str) -> List[Failure]:
    out: List[Failure] = []
    cur: Optional[Failure] = None
    for line in output.splitlines():
        h = _JEST_HEADER.match(line)
        if h and "›" in h.group("test"):
            cur = Failure(test=h.group("test").split("›")[-1].strip())
            out.append(cur)
            continue
        if cur is None:
            continue
        e = _JEST_EXPECTED.match(line)
        if e and cur.expected is None:
            cur.expected = e.group("v")
            continue
        r = _JEST_RECEIVED.match(line)
        if r and cur.actual is None:
            cur.actual = r.group("v")
            continue
        a = _JEST_AT.search(line)
        if a and cur.location is None and ".spec." in a.group("loc"):
            cur.location = a.group("loc")
    return out


# --- cargo test -------------------------------------------------------------

_CARGO_PANIC = re.compile(
    r"^thread '(?P<test>[^']+)'.*panicked at (?P<loc>[^:]+:\d+:\d+):"
)
_CARGO_LEFT = re.compile(r"^\s*left:\s*(?P<v>.+?)\s*$")
_CARGO_RIGHT = re.compile(r"^\s*right:\s*(?P<v>.+?)\s*$")
_CARGO_BARE = re.compile(r"^assertion failed: (?P<expr>.+)$")


def _parse_cargo(output: str) -> List[Failure]:
    out: List[Failure] = []
    lines = output.splitlines()
    for i, line in enumerate(lines):
        m = _CARGO_PANIC.match(line)
        if not m:
            continue
        f = Failure(test=m.group("test"), location=m.group("loc"))
        window = lines[i + 1 : i + 6]
        for w in window:
            left = _CARGO_LEFT.match(w)
            if left:
                f.actual = left.group("v")  # cargo prints `left: <actual>`
            right = _CARGO_RIGHT.match(w)
            if right:
                f.expected = right.group("v")  # `right: <expected>`
        if f.expected is None and f.actual is None:
            bare = next(
                (_CARGO_BARE.match(w) for w in window if _CARGO_BARE.match(w)), None
            )
            f.message = bare.group(0) if bare else (window[0].strip() or None)
        out.append(f)
    return out


# --- swift XCTest -----------------------------------------------------------

# `<file>:<line>: error: -[Suite testName] : XCTAssertEqual failed: ("2") is not
# equal to ("3")` — the one line carrying test, location, and the assertion.
_XCTEST = re.compile(
    r"^(?P<file>.+?):(?P<line>\d+): error: -\[\S+ (?P<test>\w+)\] : (?P<detail>.+)$"
)
_XCTEST_EQ = re.compile(r'\("(?P<actual>.*)"\) is not equal to \("(?P<expected>.*)"\)')


def _parse_xctest(output: str) -> List[Failure]:
    out: List[Failure] = []
    for line in output.splitlines():
        m = _XCTEST.match(line)
        if not m:
            continue
        f = Failure(
            test=m.group("test"), location=f"{m.group('file')}:{m.group('line')}"
        )
        detail = m.group("detail").strip()
        eq = _XCTEST_EQ.search(detail)
        if eq:
            f.actual = eq.group("actual")
            f.expected = eq.group("expected")
        else:
            f.message = detail
        out.append(f)
    return out


# --- dotnet test / xUnit ----------------------------------------------------

_DOTNET_FAILED = re.compile(r"^\s*Failed\s+(?P<test>[\w.+]+)\s+\[")
_DOTNET_EXPECTED = re.compile(r"^\s*Expected:\s*(?P<v>.+?)\s*$")
_DOTNET_ACTUAL = re.compile(r"^\s*Actual:\s*(?P<v>.+?)\s*$")
_DOTNET_AT = re.compile(r"\bat .+ in (?P<loc>.+:line \d+)")


def _parse_dotnet(output: str) -> List[Failure]:
    out: List[Failure] = []
    lines = output.splitlines()
    for i, line in enumerate(lines):
        m = _DOTNET_FAILED.match(line)
        if not m:
            continue
        f = Failure(test=m.group("test").split(".")[-1])
        for w in lines[i + 1 : i + 12]:
            if _DOTNET_FAILED.match(w):
                break
            e = _DOTNET_EXPECTED.match(w)
            if e and f.expected is None:
                f.expected = e.group("v")
            a = _DOTNET_ACTUAL.match(w)
            if a and f.actual is None:
                f.actual = a.group("v")
            loc = _DOTNET_AT.search(w)
            if loc and f.location is None:
                f.location = loc.group("loc")
        out.append(f)
    return out


# --- vitest -----------------------------------------------------------------

_VITEST_FAIL = re.compile(r"^\s*FAIL\s+(?P<file>\S+)\s+>\s+(?P<test>.+?)\s*$")
_VITEST_TOBE = re.compile(
    r"expected (?P<actual>.+?) to (?:be|equal|deeply equal) (?P<expected>.+?)(?: //|$)"
)
_VITEST_AT = re.compile(r"❯\s+(?P<loc>\S+:\d+:\d+)")


def _parse_vitest(output: str) -> List[Failure]:
    out: List[Failure] = []
    cur: Optional[Failure] = None
    for line in output.splitlines():
        h = _VITEST_FAIL.match(line)
        if h:
            cur = Failure(test=h.group("test"))
            out.append(cur)
            continue
        if cur is None:
            continue
        tb = _VITEST_TOBE.search(line)
        if tb and cur.expected is None:
            cur.actual = tb.group("actual")
            cur.expected = tb.group("expected")
        at = _VITEST_AT.search(line)
        if at and cur.location is None:
            loc = at.group("loc")
            if ".test." in loc or ".spec." in loc:
                cur.location = loc
    return out


# --- go test ----------------------------------------------------------------

# `--- FAIL: TestName (0.00s)` marks a failed test; the name follows.
_GO_FAIL = re.compile(r"^--- FAIL: (?P<test>\S+) \(")
# A boundary between tests: `=== RUN`, `--- FAIL:`, `--- PASS:`, or the summary.
_GO_BOUNDARY = re.compile(
    r"^(?:=== RUN|=== PAUSE|=== CONT|--- (?:FAIL|PASS|SKIP)|PASS|FAIL|ok|---)"
)
# A `file_test.go:12: got 5, want 3` detail line: location plus a want/got pair.
_GO_DETAIL = re.compile(r"^\s+(?P<loc>\w+_test\.go:\d+):\s*(?P<detail>.+?)\s*$")
_GO_WANTGOT = re.compile(r"got (?P<actual>.+?),\s*want (?P<expected>.+?)\s*$")
# testify renders `Error:  Not equal:  expected: 3  actual: 5`.
_GO_TESTIFY = re.compile(
    r"expected:\s*(?P<expected>.+?)\s+actual:\s*(?P<actual>.+?)\s*$"
)


def _parse_gotest(output: str) -> List[Failure]:
    out: List[Failure] = []
    lines = output.splitlines()
    for i, line in enumerate(lines):
        m = _GO_FAIL.match(line)
        if not m:
            continue
        f = Failure(test=m.group("test"))
        # go prints `t.Errorf` details indented, split around the FAIL header:
        # some before it (back to this test's `=== RUN`), some after (testify).
        window: List[str] = []
        for w in reversed(lines[:i]):
            if _GO_BOUNDARY.match(w):
                break
            window.append(w)
        for w in lines[i + 1 :]:
            if _GO_BOUNDARY.match(w) or not w.startswith((" ", "\t")):
                break
            window.append(w)
        for w in window:
            d = _GO_DETAIL.match(w)
            if d and f.location is None:
                f.location = d.group("loc")
            detail = d.group("detail") if d else w
            wg = _GO_WANTGOT.search(detail)
            if wg and f.expected is None:
                f.actual = wg.group("actual").strip()  # go convention: `got <actual>`
                f.expected = wg.group("expected").strip()  # `want <expected>`
                continue
            ta = _GO_TESTIFY.search(detail)
            if ta and f.expected is None:
                f.expected = ta.group("expected").strip()
                f.actual = ta.group("actual").strip()
        if f.expected is None and f.actual is None and f.location is not None:
            # Located but no want/got pair: carry the raw detail as the message.
            first = next(
                (_GO_DETAIL.match(w) for w in window if _GO_DETAIL.match(w)), None
            )
            f.message = first.group("detail").strip() if first else None
        out.append(f)
    return out


# --- junit / gradle (Java + Kotlin) -----------------------------------------

# gradle prints `TestClass > testMethod() FAILED` as the failed-test header.
_JUNIT_FAILED = re.compile(r"^\s*(?:\S+ > )?(?P<test>\S+?)(?:\(\))? FAILED\s*$")
# `expected: <3> but was: <5>` (JUnit5) / `expected:<3> but was:<5>` (JUnit4);
# angle brackets optional.
_JUNIT_EXPECTED = re.compile(
    r"expected:\s*<?(?P<expected>.+?)>? but was:\s*<?(?P<actual>.+?)>?\s*$"
)
# `at TestClass.testMethod(TestClass.java:12)` — the stack frame with location.
_JUNIT_AT = re.compile(r"\bat .+\((?P<loc>[\w./]+\.(?:java|kt):\d+)\)")


def _parse_junit(output: str) -> List[Failure]:
    out: List[Failure] = []
    lines = output.splitlines()
    for i, line in enumerate(lines):
        m = _JUNIT_FAILED.match(line)
        if not m:
            continue
        f = Failure(test=m.group("test").split(".")[-1])
        for w in lines[i + 1 :]:
            if _JUNIT_FAILED.match(w):
                break
            e = _JUNIT_EXPECTED.search(w)
            if e and f.expected is None:
                f.expected = e.group("expected").strip()
                f.actual = e.group("actual").strip()
            loc = _JUNIT_AT.search(w)
            if loc and f.location is None:
                f.location = loc.group("loc")
        out.append(f)
    return out


_RUNNERS = {
    "pytest": _parse_pytest,
    "jest": _parse_jest,
    "cargo": _parse_cargo,
    "xctest": _parse_xctest,
    "dotnet": _parse_dotnet,
    "vitest": _parse_vitest,
    "gotest": _parse_gotest,
    "junit": _parse_junit,
}


def _detect_runner(output: str) -> Optional[str]:
    if "panicked at" in output or re.search(
        r"^test result: (?:ok|FAILED)", output, re.M
    ):
        return "cargo"
    if "XCTAssert" in output or "Test Case '-[" in output:
        return "xctest"
    if re.search(r"^\s*Failed\s+[\w.]+\s+\[", output, re.M) and "Actual:" in output:
        return "dotnet"
    if "❯" in output and re.search(r"^\s*FAIL\s+\S+\s+>", output, re.M):
        return "vitest"
    if re.search(r"^● ", output, re.M) or (
        "Expected:" in output and "Received:" in output
    ):
        return "jest"
    if re.search(r"^--- FAIL: ", output, re.M):
        return "gotest"
    if "but was:" in output or (
        re.search(r"\bFAILED\b", output, re.M) and "at " in output
    ):
        return "junit"
    if re.search(r"^FAILED \S+::", output, re.M) or "\nE   " in output:
        return "pytest"
    return None


def extract_failures(output: str, language: Optional[str] = None) -> List[Failure]:
    """Parse runner output into failing-test records. Empty on unrecognized output."""
    if not output:
        return []
    runner = {
        "python": "pytest",
        "javascript": "jest",
        "typescript": "jest",
        "rust": "cargo",
        "swift": "xctest",
        "csharp": "dotnet",
        "go": "gotest",
        "java": "junit",
        "kotlin": "junit",
    }.get((language or "").lower())
    if runner is None or runner not in _RUNNERS or not _RUNNERS[runner](output):
        runner = _detect_runner(output)
    if runner is None:
        return []
    return _RUNNERS[runner](output)


def _cap(value: Optional[str], limit: int = 320) -> Optional[str]:
    """Keep a long value's head and tail (where a diff usually shows) and elide
    the middle, so a huge expected/actual can't dominate the retry context."""
    if value is None or len(value) <= limit:
        return value
    head = limit * 2 // 3
    return f"{value[:head]} …({len(value) - limit} chars elided)… {value[-(limit - head) :]}"


def render_failure_view(failures: List[Failure], max_failures: int = 5) -> str:
    """Render the exact failing assertions as a tight, lead-with-truth block.

    Empty string when there is nothing structured to show, so the caller keeps
    its existing compressed error context unchanged.
    """
    if not failures:
        return ""
    shown = failures[:max_failures]
    lines = [f"{len(failures)} failing test(s); the exact assertions:"]
    for f in shown:
        loc = f" ({f.location})" if f.location else ""
        lines.append(f"- {f.test}{loc}")
        if f.expected is not None or f.actual is not None:
            lines.append(f"    expected: {_cap(f.expected)}")
            lines.append(f"    actual:   {_cap(f.actual)}")
        elif f.message:
            lines.append(f"    {_cap(f.message)}")
    if len(failures) > max_failures:
        lines.append(f"  (+{len(failures) - max_failures} more failing the same way)")
    return "\n".join(lines)
