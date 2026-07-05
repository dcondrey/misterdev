"""Shared routing for an INDEPENDENT reviewer/judge model.

The orchestrator runs several LLM judgments that should not share the generating
model's blind spots: the edit-time adversarial critic, the post-build
goal-completion check, and the LLM acceptance judge. Each wants the same thing —
run ``generate_code`` on a configured independent model when the client can
switch to it, otherwise fall back to the generator's own model and make the
weaker independence visible rather than silent. This centralizes that so the
three call sites don't each re-implement it.
"""

from typing import Callable, Optional

from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)


def generate_independent(
    llm_client, prompt: str, system: str = "", *, model: Optional[str] = None
) -> str:
    """Run ``generate_code`` on ``model`` when the client can switch to it.

    Routes through the client's ``with_model`` context manager for a configured
    independent model; otherwise (no model, or a client without ``with_model``)
    runs on the client's own model. Returns the model's text (never None).
    """
    if model and hasattr(llm_client, "with_model"):
        with llm_client.with_model(model):
            return llm_client.generate_code(prompt, system) or ""
    return llm_client.generate_code(prompt, system) or ""


def build_independent_call(
    llm_client, system: str, model: Optional[str], role: str
) -> Optional[Callable[[str], str]]:
    """Build a ``call(prompt) -> str`` bound to an independent model, or None.

    Returns None when ``llm_client`` can't generate text (so the caller SKIPs).
    Otherwise returns a closure that runs each prompt through ``model`` (when the
    client supports ``with_model``) or the generator's own model. The
    independence level is logged ONCE here, at build time — a warning when a
    model is set but the client can't switch, an info when none is configured —
    so the weaker case is never silently assumed. ``role`` names the judge for
    the log. No network is touched until the returned closure is invoked.
    """
    if llm_client is None or not hasattr(llm_client, "generate_code"):
        return None

    can_switch = hasattr(llm_client, "with_model")
    if model and not can_switch:
        logger.warning(
            f"{role}: an independent model is set but the client cannot switch "
            f"models; running on the generator's own model (weaker independence)."
        )
    elif not model:
        logger.info(
            f"{role}: no independent model configured; running on the generator's "
            f"own model (weaker independence — set one for a true second "
            f"component)."
        )

    effective = model if (model and can_switch) else None

    def _call(prompt: str) -> str:
        return generate_independent(llm_client, prompt, system, model=effective)

    return _call
