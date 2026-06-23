import os
import time

import pytest

import my_project_orchestrator.core.web_verify as web_verify
from my_project_orchestrator.core.web_verify import (
    GREEN,
    RED,
    SKIP,
    WebResult,
    run_web_gate,
    _byte_diff_fraction,
    _image_diff_fraction,
)


# --- Fakes ------------------------------------------------------------------
#
# A fake Playwright sync API so the gate's plumbing (navigate, console capture,
# checks, screenshot evidence, teardown) is exercised end-to-end WITHOUT a real
# browser or network. _playwright_sync() is monkeypatched to return this.


class _FakeConsoleMsg:
    def __init__(self, msg_type, text):
        self.type = msg_type
        self.text = text


class _FakePage:
    def __init__(
        self,
        *,
        html="<html><body>hello</body></html>",
        selectors=(),
        console_errors=(),
        screenshot_bytes=b"PNG-A",
        axe_violations=None,
    ):
        self._html = html
        self._selectors = set(selectors)
        self._console_errors = list(console_errors)
        self._screenshot_bytes = screenshot_bytes
        self._axe_violations = axe_violations
        self._handlers = {}

    def on(self, event, handler):
        self._handlers.setdefault(event, []).append(handler)

    def goto(self, url, timeout=None):
        # Fire any queued console errors as the page "loads".
        for text in self._console_errors:
            for h in self._handlers.get("console", []):
                h(_FakeConsoleMsg("error", text))

    def query_selector(self, selector):
        return object() if selector in self._selectors else None

    def content(self):
        return self._html

    def screenshot(self, path=None, full_page=False):
        if path is not None:
            with open(path, "wb") as f:
                f.write(self._screenshot_bytes)
            return None
        return self._screenshot_bytes

    def add_script_tag(self, url=None):
        if self._axe_violations is None:
            raise RuntimeError("axe unavailable")

    def evaluate(self, _script):
        return self._axe_violations or []


class _FakeBrowser:
    def __init__(self, page):
        self._page = page
        self.closed = False

    def new_page(self):
        return self._page

    def close(self):
        self.closed = True


class _FakeChromium:
    def __init__(self, browser):
        self._browser = browser

    def launch(self, headless=True):
        return self._browser


class _FakePW:
    def __init__(self, page):
        self.chromium = _FakeChromium(_FakeBrowser(page))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_pw(monkeypatch, page):
    monkeypatch.setattr(web_verify, "_playwright_sync", lambda: lambda: _FakePW(page))


# --- SKIP semantics ---------------------------------------------------------


def test_skip_when_no_config(tmp_path):
    assert run_web_gate(tmp_path, None).status == SKIP
    assert run_web_gate(tmp_path, {}).status == SKIP


def test_skip_when_no_url(tmp_path):
    res = run_web_gate(tmp_path, {"checks": ["text:hi"]})
    assert res.status == SKIP
    assert res.skipped and not res.passed


def test_skip_when_playwright_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(web_verify, "_playwright_sync", lambda: None)
    res = run_web_gate(tmp_path, {"url": "http://x", "checks": ["text:hi"]})
    assert res.status == SKIP
    assert "playwright" in res.reason


# --- GREEN ------------------------------------------------------------------


def test_green_when_all_checks_pass(monkeypatch, tmp_path):
    page = _FakePage(
        html="<html><body><h1>Welcome</h1></body></html>",
        selectors=("h1",),
        axe_violations=[],
    )
    _patch_pw(monkeypatch, page)
    res = run_web_gate(
        tmp_path,
        {
            "url": "http://x",
            "baseline_dir": str(tmp_path / "shots"),
            "checks": ["dom:h1", "text:Welcome", "no-console-errors", "axe"],
            "timeout": 10,
        },
    )
    assert res.status == GREEN
    assert res.passed
    # Screenshot evidence written.
    assert os.path.exists(res.evidence)


def test_screenshot_seed_then_match(monkeypatch, tmp_path):
    shots = str(tmp_path / "shots")
    page = _FakePage(screenshot_bytes=b"PNGDATA-IDENTICAL")
    _patch_pw(monkeypatch, page)
    cfg = {
        "url": "http://x",
        "baseline_dir": shots,
        "checks": ["screenshot"],
        "timeout": 10,
    }
    first = run_web_gate(tmp_path, cfg)
    # First run seeds baseline; seeded is non-failing.
    assert first.status == GREEN
    # Second run with identical bytes diffs at ~0 -> GREEN.
    second = run_web_gate(tmp_path, cfg)
    assert second.status == GREEN


# --- RED --------------------------------------------------------------------


def test_red_when_console_error(monkeypatch, tmp_path):
    page = _FakePage(console_errors=["TypeError: boom"], axe_violations=[])
    _patch_pw(monkeypatch, page)
    res = run_web_gate(
        tmp_path,
        {
            "url": "http://x",
            "baseline_dir": str(tmp_path / "s"),
            "checks": ["no-console-errors"],
            "timeout": 10,
        },
    )
    assert res.status == RED
    assert "console error" in res.reason


def test_red_when_dom_missing(monkeypatch, tmp_path):
    page = _FakePage(selectors=())
    _patch_pw(monkeypatch, page)
    res = run_web_gate(
        tmp_path,
        {
            "url": "http://x",
            "baseline_dir": str(tmp_path / "s"),
            "checks": ["dom:#nope"],
            "timeout": 10,
        },
    )
    assert res.status == RED
    assert "not found" in res.reason


def test_red_when_text_missing(monkeypatch, tmp_path):
    page = _FakePage(html="<html><body>other</body></html>")
    _patch_pw(monkeypatch, page)
    res = run_web_gate(
        tmp_path,
        {
            "url": "http://x",
            "baseline_dir": str(tmp_path / "s"),
            "checks": ["text:ABSENT"],
            "timeout": 10,
        },
    )
    assert res.status == RED


def test_red_when_axe_violation(monkeypatch, tmp_path):
    page = _FakePage(axe_violations=["color-contrast", "image-alt"])
    _patch_pw(monkeypatch, page)
    res = run_web_gate(
        tmp_path,
        {
            "url": "http://x",
            "baseline_dir": str(tmp_path / "s"),
            "checks": ["axe"],
            "timeout": 10,
        },
    )
    assert res.status == RED
    assert "violation" in res.reason


def test_red_when_screenshot_diff_exceeds_threshold(monkeypatch, tmp_path):
    shots = tmp_path / "shots"
    shots.mkdir()
    # Seed a baseline that differs sharply from what the page will render.
    (shots / "web_verify_baseline.png").write_bytes(b"A" * 100)
    page = _FakePage(screenshot_bytes=b"B" * 100)
    _patch_pw(monkeypatch, page)
    res = run_web_gate(
        tmp_path,
        {
            "url": "http://x",
            "baseline_dir": str(shots),
            "checks": ["screenshot"],
            "threshold": 0.01,
            "timeout": 10,
        },
    )
    assert res.status == RED
    assert "diff" in res.reason


# --- never blocks (hard timeout) -------------------------------------------


def test_hanging_browser_returns_within_timeout(monkeypatch, tmp_path):
    # A browser whose context manager blocks forever must be abandoned by the
    # hard outer join, not block the caller.
    class _WedgePW:
        def __enter__(self):
            time.sleep(3600)

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(web_verify, "_playwright_sync", lambda: lambda: _WedgePW())
    start = time.monotonic()
    res = run_web_gate(tmp_path, {"url": "http://x", "checks": [], "timeout": 1})
    elapsed = time.monotonic() - start
    assert elapsed < 20
    assert res.status == SKIP
    assert not res.passed


def test_browser_error_is_skip_not_crash(monkeypatch, tmp_path):
    def _boom():
        def _factory():
            raise RuntimeError("launch failed")

        return _factory

    monkeypatch.setattr(web_verify, "_playwright_sync", _boom)
    res = run_web_gate(tmp_path, {"url": "http://x", "checks": [], "timeout": 5})
    assert res.status == SKIP
    assert "error" in res.reason


# --- helpers / result object ------------------------------------------------


def test_byte_diff_fraction():
    assert _byte_diff_fraction(b"", b"") == 0.0
    assert _byte_diff_fraction(b"AAAA", b"AAAA") == 0.0
    assert _byte_diff_fraction(b"AAAA", b"AAAB") == 0.25
    assert _byte_diff_fraction(b"AA", b"AAAA") == 0.5  # length mismatch counts


def test_image_diff_identical_bytes_is_zero():
    # Pillow can't open these bytes -> falls back to byte diff (identical = 0).
    assert _image_diff_fraction(b"not-an-image", b"not-an-image") == 0.0


def test_web_result_repr_and_flags():
    r = WebResult(GREEN, evidence="/x.png")
    assert r.passed and not r.skipped
    assert "green" in repr(r)


# --- live integration (opt-in) ----------------------------------------------


def test_live_web_runs_or_skips(tmp_path):
    # Opportunistic live integration: drive a real headless Chromium against a
    # data: URL. Gated behind RUN_WEB_INTEGRATION so a real browser is never
    # required by the normal suite; the timeout guarantees it can't hang.
    if not os.environ.get("RUN_WEB_INTEGRATION"):
        pytest.skip("set RUN_WEB_INTEGRATION=1 to exercise a real browser")
    res = run_web_gate(
        tmp_path,
        {
            "url": "data:text/html,<html><body><h1>Hi</h1></body></html>",
            "baseline_dir": str(tmp_path / "shots"),
            "checks": ["dom:h1", "text:Hi", "no-console-errors"],
            "timeout": 30,
        },
    )
    if res.status == SKIP:
        pytest.skip(f"no browser available: {res.reason}")
    assert res.status == GREEN
    assert os.path.exists(res.evidence)
