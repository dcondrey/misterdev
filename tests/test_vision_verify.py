import os
import time
import types

import pytest

from misterdev.core.verification.vision_verify import (
    GREEN,
    RED,
    SKIP,
    VisionResult,
    run_vision_gate,
    _default_vlm_call,
    _parse_verdict,
)


def _shot(tmp_path):
    p = tmp_path / "shot.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n-fake-bytes")
    return p


def test_default_vlm_call_uses_raw_client_not_with_model(tmp_path):
    # Regression: with_model is a context manager, not a client factory. The
    # default call must use llm_client.client directly (and pass model=). A
    # client with a real .client.chat.completions.create must produce a verdict.
    captured = {}

    class _Completions:
        def create(self, **kw):
            captured.update(kw)
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content="YES, a login form.")
                    )
                ]
            )

    def _boom_with_model(_m):
        raise AssertionError("with_model must not be called as a client factory")

    client = types.SimpleNamespace(
        client=types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=_Completions())
        ),
        model="base/model",
        with_model=_boom_with_model,
    )
    shot = _shot(tmp_path)
    res = run_vision_gate(
        tmp_path,
        {"capture": str(shot), "assert": "a login form", "model": "vendor/vision"},
        llm_client=client,
    )
    assert res.status == GREEN
    assert captured["model"] == "vendor/vision"  # explicit model selection used


def test_default_vlm_call_none_without_client():
    assert _default_vlm_call(None, "m") is None


# --- SKIP semantics ---------------------------------------------------------


def test_skip_when_no_config(tmp_path):
    assert run_vision_gate(tmp_path, None).status == SKIP
    assert run_vision_gate(tmp_path, {}).status == SKIP


def test_skip_when_no_capture(tmp_path):
    res = run_vision_gate(tmp_path, {"assert": "looks fine"})
    assert res.status == SKIP


def test_skip_when_no_assert(tmp_path):
    shot = _shot(tmp_path)
    res = run_vision_gate(tmp_path, {"capture": str(shot)})
    assert res.status == SKIP
    assert "assert" in res.reason


def test_skip_when_capture_missing(tmp_path):
    res = run_vision_gate(tmp_path, {"capture": "does_not_exist.png", "assert": "x"})
    assert res.status == SKIP
    assert "not found" in res.reason


def test_skip_when_no_model_or_client(tmp_path):
    shot = _shot(tmp_path)
    res = run_vision_gate(tmp_path, {"capture": str(shot), "assert": "x"})
    assert res.status == SKIP
    assert "no vision model" in res.reason


# --- GREEN / RED ------------------------------------------------------------


def test_green_on_affirm(tmp_path):
    shot = _shot(tmp_path)
    res = run_vision_gate(
        tmp_path,
        {"capture": str(shot), "assert": "shows a chart", "timeout": 10},
        vlm_call=lambda prompt, img: "YES\nThe chart is clearly visible.",
    )
    assert res.status == GREEN
    assert res.passed
    assert "YES" in res.verdict


def test_red_on_deny(tmp_path):
    shot = _shot(tmp_path)
    res = run_vision_gate(
        tmp_path,
        {"capture": str(shot), "assert": "shows a chart", "timeout": 10},
        vlm_call=lambda prompt, img: "NO\nThe page is blank.",
    )
    assert res.status == RED
    assert not res.passed
    assert "blank" in res.reason


def test_skip_on_unparseable_verdict(tmp_path):
    shot = _shot(tmp_path)
    res = run_vision_gate(
        tmp_path,
        {"capture": str(shot), "assert": "x", "timeout": 10},
        vlm_call=lambda prompt, img: "I am not sure about this image.",
    )
    assert res.status == SKIP
    assert "unparseable" in res.reason


def test_call_receives_prompt_and_b64_image(tmp_path):
    shot = _shot(tmp_path)
    seen = {}

    def _call(prompt, img):
        seen["prompt"] = prompt
        seen["img"] = img
        return "yes ok"

    run_vision_gate(
        tmp_path,
        {"capture": str(shot), "assert": "the header is red", "timeout": 10},
        vlm_call=_call,
    )
    assert "the header is red" in seen["prompt"]
    # Image is base64-encoded (no raw PNG magic bytes leak through).
    assert seen["img"] and "PNG" not in seen["img"][:8]


# --- never blocks / errors --------------------------------------------------


def test_hanging_model_returns_within_timeout(tmp_path):
    shot = _shot(tmp_path)

    def _slow(prompt, img):
        time.sleep(3600)
        return "yes"

    start = time.monotonic()
    res = run_vision_gate(
        tmp_path,
        {"capture": str(shot), "assert": "x", "timeout": 0.3},
        vlm_call=_slow,
    )
    assert time.monotonic() - start < 10
    assert res.status == SKIP
    assert "timed out" in res.reason
    assert not res.passed


def test_model_error_is_skip_not_crash(tmp_path):
    shot = _shot(tmp_path)

    def _boom(prompt, img):
        raise RuntimeError("model unreachable")

    res = run_vision_gate(
        tmp_path,
        {"capture": str(shot), "assert": "x", "timeout": 5},
        vlm_call=_boom,
    )
    assert res.status == SKIP
    assert "error" in res.reason


def test_no_client_default_call_is_none(tmp_path):
    # With neither a vlm_call nor an llm_client, the default builder yields no
    # call and the gate skips (no network ever touched).
    shot = _shot(tmp_path)
    res = run_vision_gate(
        tmp_path, {"capture": str(shot), "assert": "x"}, llm_client=None
    )
    assert res.status == SKIP


# --- verdict parsing / result object ----------------------------------------


def test_parse_verdict():
    assert _parse_verdict("YES, it does") is True
    assert _parse_verdict("yes\nreason") is True
    assert _parse_verdict("NO\nbecause...") is False
    assert _parse_verdict("false") is False
    # 'no' inside a reason must not flip an affirmation.
    assert _parse_verdict("YES, there is no error") is True
    assert _parse_verdict("maybe") is None
    assert _parse_verdict("") is None


def test_vision_result_repr_and_flags():
    r = VisionResult(GREEN, verdict="YES")
    assert r.passed and not r.skipped
    assert "green" in repr(r)


# --- default client call path -----------------------------------------------


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletions:
    def __init__(self, content):
        self._content = content
        self.seen = {}

    def create(self, model=None, messages=None):
        self.seen["model"] = model
        self.seen["messages"] = messages
        return type("R", (), {"choices": [_FakeChoice(self._content)]})()


class _FakeChat:
    def __init__(self, content):
        self.completions = _FakeCompletions(content)


class _FakeRaw:
    def __init__(self, content):
        self.chat = _FakeChat(content)


class _FakeClient:
    def __init__(self, content):
        self.client = _FakeRaw(content)
        self.model = "vision-model"

    def with_model(self, model):
        return self


def test_default_call_uses_client_multimodal(tmp_path):
    shot = _shot(tmp_path)
    client = _FakeClient("YES\nlooks right")
    res = run_vision_gate(
        tmp_path,
        {"capture": str(shot), "assert": "ok", "timeout": 10},
        llm_client=client,
    )
    assert res.status == GREEN
    msgs = client.client.chat.completions.seen["messages"]
    parts = msgs[0]["content"]
    # Multimodal: a text part and an image_url part with a base64 data URL.
    assert any(p.get("type") == "text" for p in parts)
    img = next(p for p in parts if p.get("type") == "image_url")
    assert img["image_url"]["url"].startswith("data:image/png;base64,")


def test_default_call_prefers_chat_multimodal(tmp_path):
    # A client exposing the first-class chat_multimodal method is used directly;
    # the raw .client SDK path must NOT be touched.
    seen = {}

    class _Client:
        def __init__(self):
            self.client = None  # would crash if the fallback path were taken
            self.model = "base/model"

        def chat_multimodal(self, prompt, image_b64, model=None):
            seen["prompt"] = prompt
            seen["image_b64"] = image_b64
            seen["model"] = model
            return "YES\nlooks right"

    shot = _shot(tmp_path)
    res = run_vision_gate(
        tmp_path,
        {"capture": str(shot), "assert": "ok", "model": "vendor/vision", "timeout": 10},
        llm_client=_Client(),
    )
    assert res.status == GREEN
    assert seen["model"] == "vendor/vision"  # explicit model forwarded
    assert "PNG" not in seen["image_b64"][:8]  # base64-encoded, not raw bytes


# --- gatekeeper integration -------------------------------------------------


def test_gatekeeper_skips_vision_when_off(tmp_path):
    from misterdev.core.verification.gatekeeper import GateKeeper

    (tmp_path / "a.py").write_text("x = 1\n")
    shot = _shot(tmp_path)
    keeper = GateKeeper(
        tmp_path,
        vision_verify=False,
        vision_client=_FakeClient("NO\nbroken"),
        runtime_config={"vision": {"capture": str(shot), "assert": "ok"}},
    )
    _success, issues, _ = keeper.run_gates({})
    assert not any("G4.8" in i for i in issues)


def test_gatekeeper_red_vision_blocks_build(tmp_path):
    from misterdev.core.verification.gatekeeper import GateKeeper

    (tmp_path / "a.py").write_text("x = 1\n")
    shot = _shot(tmp_path)
    keeper = GateKeeper(
        tmp_path,
        vision_verify=True,
        vision_client=_FakeClient("NO\nthe layout is broken"),
        runtime_config={"vision": {"capture": str(shot), "assert": "ok", "timeout": 5}},
    )
    success, issues, _ = keeper.run_gates({})
    assert not success
    assert any("G4.8" in i for i in issues)


def test_gatekeeper_green_vision_passes(tmp_path):
    from misterdev.core.verification.gatekeeper import GateKeeper

    (tmp_path / "a.py").write_text("x = 1\n")
    shot = _shot(tmp_path)
    keeper = GateKeeper(
        tmp_path,
        vision_verify=True,
        vision_client=_FakeClient("YES\nlooks good"),
        runtime_config={"vision": {"capture": str(shot), "assert": "ok", "timeout": 5}},
    )
    _success, issues, _ = keeper.run_gates({})
    assert not any("G4.8" in i for i in issues)


# --- live integration (opt-in) ----------------------------------------------


_LOGIN_FORM_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>Sign in</title></head><body style="font-family:sans-serif;padding:40px">
<h1>Sign in</h1>
<form>
  <div><label>Username<br><input type="text" name="username"></label></div>
  <div style="margin-top:12px"><label>Password<br>
    <input type="password" name="password"></label></div>
  <div style="margin-top:16px"><button type="submit">Log in</button></div>
</form>
</body></html>"""


def _render_login_screenshot(tmp_path):
    """Render the login form to a real PNG via headless Chromium.

    Returns the screenshot path, or None when Playwright/the browser is absent
    so the caller can skip rather than fail.
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return None
    page_html = tmp_path / "login.html"
    page_html.write_text(_LOGIN_FORM_HTML)
    shot = tmp_path / "login.png"
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(page_html.as_uri(), timeout=15000)
                page.screenshot(path=str(shot), full_page=True)
            finally:
                browser.close()
    except Exception:
        return None
    return shot if shot.is_file() else None


def test_vision_integration_runs_or_skips(tmp_path):
    # Opportunistic live integration against a real VLM (OpenRouter gpt-4o-mini)
    # over a real screenshot rendered by headless Chromium. Gated behind
    # RUN_VISION_INTEGRATION so the normal suite never needs a model, key, or
    # browser; the gate's own hard timeout guarantees it can't hang. Skips
    # cleanly when the key or the browser is absent.
    if not os.environ.get("RUN_VISION_INTEGRATION"):
        pytest.skip("set RUN_VISION_INTEGRATION=1 to exercise a real VLM")
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")

    shot = _render_login_screenshot(tmp_path)
    if shot is None:
        pytest.skip("playwright/chromium unavailable to render the screenshot")

    from misterdev.llm.client import OpenRouterLLMClient

    client = OpenRouterLLMClient(
        {
            "llm": {
                "provider": "openrouter",
                "model": "openai/gpt-4o-mini",
                "api_key_env_var": "OPENROUTER_API_KEY",
                "temperature": 0.0,
            },
            "build": {"budget": 5.0},
        }
    )

    match = run_vision_gate(
        tmp_path,
        {
            "capture": str(shot),
            "assert": "a login form with a password field and a submit button",
            "model": "openai/gpt-4o-mini",
            "timeout": 60,
        },
        llm_client=client,
    )
    assert match.status == GREEN, f"expected GREEN, got {match!r} ({match.verdict!r})"

    mismatch = run_vision_gate(
        tmp_path,
        {
            "capture": str(shot),
            "assert": "a bar chart of quarterly revenue",
            "model": "openai/gpt-4o-mini",
            "timeout": 60,
        },
        llm_client=client,
    )
    assert mismatch.status == RED, (
        f"expected RED, got {mismatch!r} ({mismatch.verdict!r})"
    )
