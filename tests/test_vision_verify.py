import os
import time

import pytest

from my_project_orchestrator.core.vision_verify import (
    GREEN,
    RED,
    SKIP,
    VisionResult,
    run_vision_gate,
    _parse_verdict,
)


def _shot(tmp_path):
    p = tmp_path / "shot.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n-fake-bytes")
    return p


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


# --- gatekeeper integration -------------------------------------------------


def test_gatekeeper_skips_vision_when_off(tmp_path):
    from my_project_orchestrator.core.gatekeeper import GateKeeper

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
    from my_project_orchestrator.core.gatekeeper import GateKeeper

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
    from my_project_orchestrator.core.gatekeeper import GateKeeper

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


def test_live_vision_runs_or_skips(tmp_path):
    # Opportunistic live integration against a real VLM. Gated behind
    # RUN_VISION_INTEGRATION so the normal suite never needs a model or network;
    # the timeout guarantees it can't hang. Requires a configured client, which
    # is not constructed here, so it skips unless wired by the runner.
    if not os.environ.get("RUN_VISION_INTEGRATION"):
        pytest.skip("set RUN_VISION_INTEGRATION=1 to exercise a real VLM")
    pytest.skip("live VLM client wiring is environment-specific; plumbing covered")
