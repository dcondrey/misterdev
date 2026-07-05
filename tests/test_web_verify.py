import os
import re
import time

import pytest

import misterdev.core.verification.web_verify as web_verify
from misterdev.core.verification.web_verify import (
    GREEN,
    RED,
    SKIP,
    WebResult,
    run_web_gate,
    _byte_diff_fraction,
    _capture_evidence,
    _image_diff_fraction,
    _record_console,
    _run_axe,
    _run_check,
    _run_screenshot_diff,
    _terminate,
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

    def inner_text(self, selector):
        # Approximate a browser's rendered text: strip tags from the stored HTML.
        return re.sub(r"<[^>]+>", "", self._html)

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


def test_text_check_ignores_substring_only_in_markup(monkeypatch, tmp_path):
    # "header" appears only as a tag name / class, never as visible text, so the
    # text: check must NOT consider it present (was a false positive vs raw HTML).
    page = _FakePage(
        html='<html><body><header class="header">Hi</header></body></html>'
    )
    _patch_pw(monkeypatch, page)
    res = run_web_gate(
        tmp_path,
        {
            "url": "http://x",
            "baseline_dir": str(tmp_path / "s"),
            "checks": ["text:header"],
            "timeout": 10,
        },
    )
    assert res.status == RED


def test_text_check_matches_rendered_text(monkeypatch, tmp_path):
    page = _FakePage(
        html='<html><body><header class="nav">Welcome</header></body></html>'
    )
    _patch_pw(monkeypatch, page)
    res = run_web_gate(
        tmp_path,
        {
            "url": "http://x",
            "baseline_dir": str(tmp_path / "s"),
            "checks": ["text:Welcome"],
            "timeout": 10,
        },
    )
    assert res.status == GREEN


def test_text_check_falls_back_to_content_when_inner_text_unavailable():
    # A page object exposing only content() (no inner_text) still works via the
    # raw-HTML fallback rather than erroring.
    class _OnlyContent:
        def content(self):
            return "visible Hi there"

    ok, _ = _run_check(_OnlyContent(), "text:Hi", [], None, 0.0)
    assert ok is True


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


def test_serve_started_and_torn_down(monkeypatch, tmp_path):
    # A real (short-lived) server process is launched, its readiness signal is
    # awaited, the fake browser runs, then the server is torn down.
    import sys

    page = _FakePage(html="<html><body>ok</body></html>")
    _patch_pw(monkeypatch, page)
    serve = (
        f"{sys.executable} -c "
        "\"import time;print('LISTENING',flush=True);time.sleep(30)\""
    )
    res = run_web_gate(
        tmp_path,
        {
            "url": "http://x",
            "serve": serve,
            "ready": "LISTENING",
            "baseline_dir": str(tmp_path / "s"),
            "checks": ["text:ok"],
            "timeout": 10,
        },
    )
    assert res.status == GREEN


def test_serve_without_ready_signal(monkeypatch, tmp_path):
    import sys

    page = _FakePage(html="<html><body>ok</body></html>")
    _patch_pw(monkeypatch, page)
    serve = f'{sys.executable} -c "import time;time.sleep(30)"'
    res = run_web_gate(
        tmp_path,
        {
            "url": "http://x",
            "serve": serve,
            "baseline_dir": str(tmp_path / "s"),
            "checks": ["text:ok"],
            "timeout": 5,
        },
    )
    assert res.status == GREEN


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


# --- gatekeeper integration -------------------------------------------------


def test_gatekeeper_skips_web_when_off(tmp_path):
    from misterdev.core.verification.gatekeeper import GateKeeper

    (tmp_path / "a.py").write_text("x = 1\n")
    # web_verify off -> gate not run even with a (would-fail) spec present.
    keeper = GateKeeper(
        tmp_path,
        web_verify=False,
        runtime_config={"web": {"url": "http://x", "checks": ["dom:#nope"]}},
    )
    success, issues, _ = keeper.run_gates({})
    assert not any("G4.7" in i for i in issues)


def test_gatekeeper_red_web_blocks_build(monkeypatch, tmp_path):
    from misterdev.core.verification.gatekeeper import GateKeeper

    (tmp_path / "a.py").write_text("x = 1\n")
    page = _FakePage(selectors=())
    _patch_pw(monkeypatch, page)
    keeper = GateKeeper(
        tmp_path,
        web_verify=True,
        runtime_config={
            "web": {
                "url": "http://x",
                "baseline_dir": str(tmp_path / "s"),
                "checks": ["dom:#missing"],
                "timeout": 5,
            }
        },
    )
    success, issues, _ = keeper.run_gates({})
    assert not success
    assert any("G4.7" in i for i in issues)


def test_gatekeeper_green_web_passes(monkeypatch, tmp_path):
    from misterdev.core.verification.gatekeeper import GateKeeper

    (tmp_path / "a.py").write_text("x = 1\n")
    page = _FakePage(html="<html><body><h1>Up</h1></body></html>", selectors=("h1",))
    _patch_pw(monkeypatch, page)
    keeper = GateKeeper(
        tmp_path,
        web_verify=True,
        runtime_config={
            "web": {
                "url": "http://x",
                "baseline_dir": str(tmp_path / "s"),
                "checks": ["dom:h1", "text:Up"],
                "timeout": 5,
            }
        },
    )
    success, issues, _ = keeper.run_gates({})
    assert not any("G4.7" in i for i in issues)


# --- helpers / result object ------------------------------------------------


def test_byte_diff_fraction():
    assert _byte_diff_fraction(b"", b"") == 0.0
    assert _byte_diff_fraction(b"AAAA", b"AAAA") == 0.0
    assert _byte_diff_fraction(b"AAAA", b"AAAB") == 0.25
    assert _byte_diff_fraction(b"AA", b"AAAA") == 0.5  # length mismatch counts


def test_image_diff_identical_bytes_is_zero():
    # Pillow can't open these bytes -> falls back to byte diff (identical = 0).
    assert _image_diff_fraction(b"not-an-image", b"not-an-image") == 0.0


def _png(color, size=(4, 4)):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_image_diff_pillow_pixel_path():
    pytest.importorskip("PIL")  # optional 'web' extra; skip cleanly when absent
    # Real PNGs exercise the Pillow per-pixel comparison (not the byte fallback).
    red = _png((255, 0, 0))
    assert _image_diff_fraction(red, red) == 0.0
    blue = _png((0, 0, 255))
    assert _image_diff_fraction(red, blue) == 1.0  # every pixel differs
    # Differing sizes are normalized via resize, not crash.
    big_blue = _png((0, 0, 255), size=(8, 8))
    assert _image_diff_fraction(red, big_blue) == 1.0


# --- internal check helpers (browser-free) ----------------------------------


def test_run_check_dom_selector_error_is_red():
    class _Boom:
        def query_selector(self, sel):
            raise RuntimeError("bad selector")

    ok, detail = _run_check(_Boom(), "dom:::", [], None, 0.0)
    assert ok is False and "selector error" in detail


def test_run_check_text_content_error_is_red():
    class _Boom:
        def content(self):
            raise RuntimeError("no content")

    ok, detail = _run_check(_Boom(), "text:hi", [], None, 0.0)
    assert ok is False and "content error" in detail


def test_run_check_unknown_check_is_red():
    ok, detail = _run_check(_FakePage(), "totally-unknown", [], None, 0.0)
    assert ok is False and detail == "unknown check"


def test_run_axe_unavailable_passes():
    # axe-core fails to inject -> no opinion (pass the individual check).
    page = _FakePage(axe_violations=None)
    ok, detail = _run_axe(page)
    assert ok is True and "unavailable" in detail


def test_run_axe_no_violations_passes():
    page = _FakePage(axe_violations=[])
    ok, detail = _run_axe(page)
    assert ok is True and detail == "no violations"


def test_run_screenshot_diff_seed_returns_none(tmp_path):
    page = _FakePage(screenshot_bytes=b"PNGDATA")
    ok, detail = _run_screenshot_diff(page, tmp_path, 0.01)
    assert ok is None and "seeded" in detail


def test_run_screenshot_diff_error_skips():
    class _Boom:
        def screenshot(self, full_page=False):
            raise RuntimeError("no shot")

    ok, detail = _run_screenshot_diff(_Boom(), None, 0.0)
    assert ok is True and "unavailable" in detail


def test_record_console_only_errors():
    sink = []
    _record_console(_FakeConsoleMsg("log", "info line"), sink)
    _record_console(_FakeConsoleMsg("error", "boom"), sink)
    assert sink == ["boom"]


def test_capture_evidence_failure_returns_empty():
    class _Boom:
        def screenshot(self, path=None, full_page=False):
            raise RuntimeError("no shot")

    # baseline_dir under a path component that is a file -> mkdir/write fails.
    assert _capture_evidence(_Boom(), web_verify.Path("/dev/null/x")) == ""


def test_terminate_none_and_dead_process_is_noop():
    _terminate(None)  # must not raise

    class _Dead:
        def poll(self):
            return 0

    _terminate(_Dead())  # already exited -> no-op


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
