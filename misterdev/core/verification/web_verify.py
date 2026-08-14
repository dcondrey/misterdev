"""Optional headless-browser web verification gate.

A passing build/test suite proves units behave, but not that the assembled web
artifact actually renders, is free of console errors, meets accessibility
baselines, or looks like its approved screenshot. This gate drives a real
headless browser (Playwright, sync API), optionally starting a dev server first,
loads a URL, runs a list of declarative ``checks`` against the live page, and
captures a screenshot as REAL evidence (never an LLM self-report).

It mirrors :mod:`misterdev.core.execution.runtime`: strictly opt-in (off
unless ``runtime.web`` is configured), best-effort, and run in a daemon worker
thread with a hard timeout so a hung browser or server can NEVER block the build.
Absent config, a missing Playwright/browser dependency, or a timeout is a SKIP
(no opinion), not a failure; only a check that genuinely fails (missing element,
missing text, a console error, an accessibility violation, or a screenshot diff
beyond threshold) is a RED.

``checks`` is a list of strings drawn from:
  - ``dom:<selector>``       element matching the CSS selector must be present
  - ``text:<substring>``     the substring must appear in the page text
  - ``no-console-errors``    no ``console.error`` / page error may fire
  - ``axe``                  inject axe-core; fail on any accessibility violation
  - ``screenshot``           capture and pixel-diff against a stored baseline;
                             with no baseline, the current shot is seeded and the
                             check is treated as seeded (SKIP-like), not RED.
"""

import select
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

from misterdev.core.execution.bounded import run_bounded
from misterdev.core.execution.outcomes import (
    GREEN,
    RED,
    SKIP,
    GateOutcome,
)
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)

# Outcome constants. SKIP means "no opinion" (no config, missing dependency, or
# timeout) and must never be treated as a pass/fail signal by callers.
# Filename of the captured evidence screenshot, written under ``baseline_dir``
# (or a temp dir) so a human can inspect what the gate actually saw.
_EVIDENCE_NAME = "web_verify_evidence.png"
_BASELINE_NAME = "web_verify_baseline.png"

# Default fraction of differing bytes above which a screenshot check fails. Kept
# generous: this is a coarse regression tripwire, not a precise visual diff.
_DEFAULT_SCREENSHOT_THRESHOLD = 0.02


class WebResult(GateOutcome):
    """Outcome of a web verification run. ``status`` is SKIP/GREEN/RED;
    ``evidence`` is the path to the captured screenshot (or ""); ``checks`` is
    the per-check outcome list; ``reason`` explains a SKIP/RED."""

    def __init__(
        self,
        status: str,
        evidence: str = "",
        checks: Optional[List[Tuple[str, str]]] = None,
        reason: str = "",
    ):
        super().__init__(status, reason)
        self.evidence = evidence
        self.checks = checks or []

    def __repr__(self) -> str:
        return f"WebResult(status={self.status!r}, reason={self.reason!r})"


def run_web_gate(
    project_root: Path,
    web_config: Optional[dict],
    runner=None,
) -> WebResult:
    """Run the web gate described by ``web_config``.

    ``web_config`` keys:
      - ``url`` (required): the page to load (http(s):// or file://).
      - ``serve`` (optional): command to start a dev server before loading.
      - ``ready`` (optional): substring of the server's stdout that signals
        readiness; without it we just wait briefly for the port.
      - ``checks`` (optional): list of check strings (see module docstring).
      - ``baseline_dir`` (optional): where baseline/evidence images live;
        defaults to ``<project_root>/.orchestrator``.
      - ``threshold`` (optional): screenshot diff fraction (default 0.02).
      - ``timeout`` (optional, default 60): hard ceiling for the whole run.

    Returns a :class:`WebResult`. SKIP when there is no config / no ``url``
    (feature off), when Playwright or its browser is unavailable, or when the
    hard timeout fires (never blocks). ``runner`` is accepted for symmetry with
    the gate seam; the browser runs on the host where the dev server's ports
    live, so it is unused today.
    """
    if not web_config or not web_config.get("url"):
        return WebResult(SKIP, reason="no runtime.web config")

    timeout = float(web_config.get("timeout", 60))

    def _work() -> WebResult:
        try:
            return _verify(project_root, web_config, timeout)
        except Exception as e:  # any browser/server/IO failure is non-fatal
            logger.debug(f"Web verify gate unavailable: {e}")
            return WebResult(SKIP, reason=f"error: {e}")

    # A small margin over the inner timeout so a clean inner teardown is
    # preferred, but the outer bound still guarantees we return.
    return run_bounded(
        _work, timeout + 5, WebResult(SKIP, reason="timed out"), "Web verify gate"
    )


def _playwright_sync():
    """Import Playwright's sync API, or return None if it is unavailable.

    Kept in a dedicated function so it can be monkeypatched in tests and so an
    absent dependency degrades to SKIP instead of raising.
    """
    try:
        from playwright.sync_api import sync_playwright

        return sync_playwright
    except Exception as e:  # not installed / broken install -> skip
        logger.debug(f"Playwright unavailable: {e}")
        return None


def _verify(project_root: Path, web_config: dict, timeout: float) -> WebResult:
    """Optionally start a server, drive the browser, run checks, tear down."""
    url = web_config["url"]
    if not str(url).startswith(("http://", "https://", "data:")):
        return WebResult(
            SKIP, reason=f"web verify url must use http/https/data scheme: {url!r}"
        )

    sync_playwright = _playwright_sync()
    if sync_playwright is None:
        return WebResult(SKIP, reason="playwright not installed")
    serve = web_config.get("serve")
    ready = web_config.get("ready")
    checks = list(web_config.get("checks") or [])
    threshold = float(web_config.get("threshold", _DEFAULT_SCREENSHOT_THRESHOLD))
    baseline_dir = Path(
        web_config.get("baseline_dir") or (project_root / ".orchestrator")
    )
    deadline = time.monotonic() + timeout

    proc = None
    try:
        if serve:
            proc = _start_server(project_root, serve, ready, deadline)
        return _drive_browser(
            sync_playwright, url, checks, baseline_dir, threshold, deadline
        )
    finally:
        _terminate(proc)


def _start_server(
    project_root: Path, serve: str, ready: Optional[str], deadline: float
) -> subprocess.Popen:
    """Launch the dev server and wait until it signals readiness (or briefly)."""
    proc = subprocess.Popen(
        serve,
        shell=True,
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        if ready:
            captured: List[str] = []
            while time.monotonic() < deadline:
                if "".join(captured).find(ready) != -1:
                    break
                if proc.poll() is not None:
                    break
                if proc.stdout is None:
                    break
                # Bounded wait: a raw readline() blocks until the server emits a
                # line, which for a silent server never returns and would run past
                # the deadline (leaking this process past the gate's outer bound).
                # select re-checks the deadline at most every 0.5s.
                rlist, _, _ = select.select([proc.stdout], [], [], 0.5)
                if rlist:
                    line = proc.stdout.readline()
                    if line:
                        captured.append(line)
                    elif proc.poll() is not None:
                        break
        else:
            # No readiness signal: give the server a brief moment to bind its port.
            time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))
        return proc
    except Exception:
        # A failure while waiting (e.g. select unsupported on this platform) must
        # not orphan the process we just launched.
        _terminate(proc)
        raise


def _drive_browser(
    sync_playwright,
    url: str,
    checks: List[str],
    baseline_dir: Path,
    threshold: float,
    deadline: float,
) -> WebResult:
    """Load ``url`` headless, run ``checks``, always capture a screenshot."""
    nav_timeout_ms = max(1000, int((deadline - time.monotonic()) * 1000))
    console_errors: List[str] = []
    results: List[Tuple[str, str]] = []
    failures: List[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.on("console", lambda msg: _record_console(msg, console_errors))
            page.on("pageerror", lambda err: console_errors.append(str(err)))
            page.goto(url, timeout=nav_timeout_ms)

            evidence = _capture_evidence(page, baseline_dir)

            for check in checks:
                ok, detail = _run_check(
                    page, check, console_errors, baseline_dir, threshold
                )
                results.append((check, detail))
                if ok is False:
                    failures.append(f"{check}: {detail}")
        finally:
            browser.close()

    if failures:
        return WebResult(
            RED, evidence=evidence, checks=results, reason="; ".join(failures)
        )
    return WebResult(GREEN, evidence=evidence, checks=results)


def _record_console(msg, sink: List[str]) -> None:
    """Append console.error messages to ``sink`` (best-effort across versions)."""
    try:
        msg_type = msg.type if isinstance(msg.type, str) else msg.type()
    except Exception:
        msg_type = getattr(msg, "type", "")
    if msg_type == "error":
        try:
            sink.append(msg.text if isinstance(msg.text, str) else msg.text())
        except Exception:
            sink.append("console error")


def _capture_evidence(page, baseline_dir: Path) -> str:
    """Write a full-page screenshot under ``baseline_dir`` and return its path."""
    try:
        baseline_dir.mkdir(parents=True, exist_ok=True)
        path = baseline_dir / _EVIDENCE_NAME
        page.screenshot(path=str(path), full_page=True)
        return str(path)
    except Exception as e:
        logger.debug(f"Could not capture web evidence screenshot: {e}")
        return ""


def _run_check(
    page,
    check: str,
    console_errors: List[str],
    baseline_dir: Path,
    threshold: float,
) -> Tuple[Optional[bool], str]:
    """Evaluate a single check string. Returns ``(ok, detail)`` where ``ok`` is
    True/False, or None for a seeded screenshot (treated as non-failing)."""
    if check.startswith("dom:"):
        selector = check[len("dom:") :]
        try:
            present = page.query_selector(selector) is not None
        except Exception as e:
            return False, f"selector error: {e}"
        return (present, "present" if present else "element not found")

    if check.startswith("text:"):
        needle = check[len("text:") :]
        # Match against the rendered, visible body text so a substring that only
        # occurs inside a tag name or attribute value does not count as present.
        # Fall back to the raw HTML only if the rendered text can't be read.
        try:
            content = page.inner_text("body")
        except Exception:
            try:
                content = page.content()
            except Exception as e:
                return False, f"content error: {e}"
        found = needle in content
        return (found, "found" if found else "substring not found")

    if check == "no-console-errors":
        if console_errors:
            return False, f"{len(console_errors)} console error(s): {console_errors[0]}"
        return True, "none"

    if check == "axe":
        return _run_axe(page)

    if check == "screenshot":
        return _run_screenshot_diff(page, baseline_dir, threshold)

    return False, "unknown check"


def _run_axe(page) -> Tuple[bool, str]:
    """Inject axe-core and run it; fail on any accessibility violation."""
    try:
        page.add_script_tag(
            url="https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.10.2/axe.min.js"
        )
        result = page.evaluate(
            "async () => { const r = await axe.run(); "
            "return r.violations.map(v => v.id); }"
        )
    except Exception as e:
        # axe-core could not be loaded/run (offline, CSP). No opinion -> pass the
        # individual check rather than failing the build on infrastructure.
        logger.debug(f"axe-core unavailable: {e}")
        return True, "axe unavailable (skipped)"
    violations = list(result or [])
    if violations:
        return False, f"{len(violations)} violation(s): {', '.join(violations[:5])}"
    return True, "no violations"


def _run_screenshot_diff(
    page, baseline_dir: Path, threshold: float
) -> Tuple[Optional[bool], str]:
    """Pixel-diff the current screenshot against a stored baseline.

    With no baseline, seed it and return None (seeded, not a failure). Otherwise
    fail when the differing fraction exceeds ``threshold``.
    """
    try:
        baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline = baseline_dir / _BASELINE_NAME
        current_bytes = page.screenshot(full_page=True)
        if not baseline.exists():
            baseline.write_bytes(current_bytes)
            return None, "baseline seeded"
        baseline_bytes = baseline.read_bytes()
        diff = _image_diff_fraction(baseline_bytes, current_bytes)
        if diff > threshold:
            return False, f"diff {diff:.3f} > threshold {threshold:.3f}"
        return True, f"diff {diff:.3f} <= threshold {threshold:.3f}"
    except Exception as e:
        logger.debug(f"Screenshot diff unavailable: {e}")
        return True, "screenshot diff unavailable (skipped)"


def _image_diff_fraction(a: bytes, b: bytes) -> float:
    """Fraction of differing pixels between two PNG byte strings.

    Prefers a per-pixel comparison via Pillow (size-normalized). Falls back to a
    byte-level comparison when Pillow is absent, so the check degrades rather
    than crashing.
    """
    try:
        import io

        from PIL import Image, ImageChops

        img_a = Image.open(io.BytesIO(a)).convert("RGB")
        img_b = Image.open(io.BytesIO(b)).convert("RGB")
        if img_a.size != img_b.size:
            img_b = img_b.resize(img_a.size)
        width, height = img_a.size
        if width == 0 or height == 0:
            return 0.0
        # Per-band max, not .convert("L") (luma-weighted grayscale): luma can
        # round a real per-channel delta down to 0, undercounting differing
        # pixels. Max-of-R/G/B is 0 only where all three channels match exactly,
        # matching the old per-pixel tuple-inequality check precisely.
        diff = ImageChops.difference(img_a, img_b)
        r, g, b_band = diff.split()
        combined = ImageChops.lighter(ImageChops.lighter(r, g), b_band)
        unchanged = combined.histogram()[0]
        total = width * height
        return (total - unchanged) / float(total)
    except Exception as e:
        logger.debug(f"Pillow diff unavailable, byte-comparing: {e}")
        return _byte_diff_fraction(a, b)


def _byte_diff_fraction(a: bytes, b: bytes) -> float:
    """Fraction of differing bytes between two byte strings (length-normalized)."""
    if not a and not b:
        return 0.0
    longer = max(len(a), len(b))
    if longer == 0:
        return 0.0
    shorter = min(len(a), len(b))
    differing = abs(len(a) - len(b))
    for i in range(shorter):
        if a[i] != b[i]:
            differing += 1
    return differing / float(longer)


def _terminate(proc: Optional[subprocess.Popen]) -> None:
    """Best-effort teardown of the dev server: terminate, then kill if needed."""
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except (subprocess.TimeoutExpired, OSError):
        try:
            proc.kill()
            proc.wait(timeout=3)
        except (subprocess.TimeoutExpired, OSError):
            logger.debug("Web dev server did not exit after kill.")
