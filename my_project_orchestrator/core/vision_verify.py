"""Optional vision verification gate.

Some acceptance criteria are visual ("the dashboard shows a chart", "the layout
is not broken") and cannot be asserted by a DOM/text check. This gate takes a
screenshot path and asks a vision-language model a yes/no question — "does this
image satisfy: <assert>?" — capturing the model's verdict and reason as
evidence.

It mirrors :mod:`my_project_orchestrator.core.runtime`: strictly opt-in (off
unless ``runtime.vision`` is configured), best-effort, and run in a daemon worker
thread with a hard timeout so a slow or unreachable model can NEVER block the
build. Absent config, no model/network, or a timeout is a SKIP (no opinion), not
a failure; only a model that affirmatively denies the assertion is a RED, and
only an affirmation is a GREEN.

Unlike the web gate (which captures objective browser evidence), this gate's
signal IS a model judgment — so it asserts a concrete image (real pixels the
build produced) rather than letting the model self-report on code it wrote.
"""

import base64
import re
from pathlib import Path
from typing import Callable, Optional

from my_project_orchestrator.core.bounded import run_bounded
from my_project_orchestrator.core.outcomes import GREEN, RED, SKIP, GateOutcome
from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)

# Outcome constants. SKIP means "no opinion" (no config, no model/network, or
# timeout) and must never be treated as a pass/fail signal by callers.
_PROMPT = (
    "You are a strict visual acceptance checker. Look at the attached screenshot "
    "and answer ONLY whether it satisfies this requirement:\n\n"
    "{assertion}\n\n"
    "Reply with 'YES' or 'NO' on the first line, then a one-sentence reason."
)

# A vision call takes the prompt text and the base64-encoded PNG and returns the
# model's text verdict. Injected in tests; defaulted to the project client path.
VlmCall = Callable[[str, str], str]


class VisionResult(GateOutcome):
    """Outcome of a vision check. ``status`` is SKIP/GREEN/RED; ``verdict`` is
    the raw model text (evidence); ``reason`` explains a SKIP/RED."""

    def __init__(self, status: str, verdict: str = "", reason: str = ""):
        super().__init__(status, reason)
        self.verdict = verdict

    def __repr__(self) -> str:
        return f"VisionResult(status={self.status!r}, reason={self.reason!r})"


def run_vision_gate(
    project_root: Path,
    vision_config: Optional[dict],
    vlm_call: Optional[VlmCall] = None,
    llm_client=None,
) -> VisionResult:
    """Run the vision gate described by ``vision_config``.

    ``vision_config`` keys:
      - ``capture`` (required): path to the screenshot to evaluate, relative to
        ``project_root`` or absolute.
      - ``assert`` (required): the requirement the image must satisfy.
      - ``model`` (optional): vision model id; forwarded to the client.
      - ``timeout`` (optional, default 60): hard ceiling for the whole run.

    The model call is performed by ``vlm_call`` when supplied (the test seam);
    otherwise a call is built from ``llm_client`` if one is provided. With
    neither a callable nor a client, the gate SKIPs (no model/network). SKIP also
    on absent config, a missing capture file, or the hard timeout (never blocks).
    """
    if not vision_config or not vision_config.get("capture"):
        return VisionResult(SKIP, reason="no runtime.vision config")
    assertion = vision_config.get("assert")
    if not assertion:
        return VisionResult(SKIP, reason="no assert in runtime.vision config")

    capture = Path(vision_config["capture"])
    if not capture.is_absolute():
        capture = project_root / capture
    if not capture.is_file():
        return VisionResult(SKIP, reason=f"capture file not found: {capture}")

    model = vision_config.get("model")
    timeout = float(vision_config.get("timeout", 60))

    call = vlm_call or _default_vlm_call(llm_client, model)
    if call is None:
        return VisionResult(SKIP, reason="no vision model available")

    def _work() -> VisionResult:
        try:
            return _verify(capture, assertion, call)
        except Exception as e:  # any model/IO failure is non-fatal -> skip
            logger.debug(f"Vision verify gate unavailable: {e}")
            return VisionResult(SKIP, reason=f"error: {e}")

    return run_bounded(
        _work, timeout, VisionResult(SKIP, reason="timed out"), "Vision verify gate"
    )


def _verify(capture: Path, assertion: str, call: VlmCall) -> VisionResult:
    """Encode the screenshot, ask the model, parse the YES/NO verdict."""
    image_b64 = base64.b64encode(capture.read_bytes()).decode("ascii")
    prompt = _PROMPT.format(assertion=assertion)
    verdict = call(prompt, image_b64) or ""
    decision = _parse_verdict(verdict)
    if decision is True:
        return VisionResult(GREEN, verdict=verdict)
    if decision is False:
        return VisionResult(RED, verdict=verdict, reason=verdict.strip())
    # Model gave no parseable yes/no -> no opinion, not a failure.
    return VisionResult(SKIP, verdict=verdict, reason="unparseable verdict")


def _parse_verdict(text: str) -> Optional[bool]:
    """True for an affirmation, False for a denial, None when unparseable.

    Matches a leading YES/NO token (the prompt asks for it on the first line) to
    avoid being fooled by the word appearing inside the reason sentence.
    """
    if not text:
        return None
    head = text.strip().splitlines()[0] if text.strip() else ""
    if re.match(r"^\s*(yes|true|pass|affirm)\b", head, re.IGNORECASE):
        return True
    if re.match(r"^\s*(no|false|fail|deny)\b", head, re.IGNORECASE):
        return False
    return None


def _default_vlm_call(llm_client, model: Optional[str]) -> Optional[VlmCall]:
    """Build a vision call from the project's LLM client, or None if unusable.

    Sends a multimodal message (text + a base64 PNG image) and returns the
    model's text. Prefers the client's first-class ``chat_multimodal`` method;
    falls back to driving the raw OpenAI-compatible ``.client`` SDK directly for
    clients that predate it. Kept tolerant of client shape so an absent/limited
    client degrades to SKIP rather than raising. No network is touched until the
    returned callable is actually invoked inside the worker thread.
    """
    if llm_client is None:
        return None

    def _call(prompt: str, image_b64: str) -> str:
        multimodal = getattr(llm_client, "chat_multimodal", None)
        if callable(multimodal):
            return multimodal(prompt, image_b64, model)
        # Fallback: drive the raw OpenAI-compatible SDK client directly.
        # with_model is a context manager (not a client factory), so we must not
        # rebind through it; the explicit ``model=`` below selects the vision
        # model.
        raw = getattr(llm_client, "client", None)
        if raw is None or not hasattr(raw, "chat"):
            raise RuntimeError("client does not expose a multimodal chat endpoint")
        resp = raw.chat.completions.create(
            model=model or getattr(llm_client, "model", None),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
        )
        return resp.choices[0].message.content or ""

    return _call
