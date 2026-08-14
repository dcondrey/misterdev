"""Self-reflection on a failed attempt (the Reflexion pattern).

When an attempt fails a gate, feeding the raw error into the next attempt often
just yields another local patch of the same symptom. This component instead asks
the model to step back and reflect: what is the ROOT CAUSE of the failure, and
what concretely DIFFERENT thing should the next attempt do? The reflection is
accumulated across attempts, so attempt N sees a running memory of why 1..N-1
failed — which breaks the "same wrong fix, retried" loop and lifts the success
rate on hard tasks generally, not on any one class of problem.

Mirrors :mod:`misterdev.core.verification.critic`: best-effort, timeout-bounded
in a daemon thread, and SKIP (return "") on no client, an empty result, or any
error — it only ever ADDS guidance to the next attempt, never blocks one.
"""

from typing import Callable, List, Optional

from misterdev.core.execution.bounded import run_bounded
from misterdev.core.verification.independent import build_independent_call
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)

# Keep a reflection short: it is guidance, not a transcript. A long reflection
# just crowds out the code context in the next attempt's budget.
_MAX_REFLECTION_CHARS = 1200
# Bound the failing output shown to the reflector so a huge log can't blow up the
# prompt (the tail carries the actual assertion/error).
_MAX_OUTPUT_CHARS = 4000

# A reflection call takes the assembled prompt and returns the model's text.
ReflectCall = Callable[[str], str]

_PROMPT = (
    "You are debugging your own failed attempt at a coding task. Step back from "
    "the specific error and reflect on the ROOT CAUSE, then state what to do "
    "differently — do not just restate the error.\n\n"
    "## Task\n{task}\n\n"
    "## What failed this attempt\n{output}\n"
    "{prior}\n"
    "Reply with at most 4 short lines:\n"
    "1. Root cause: the underlying reason this failed (not the surface symptom).\n"
    "2. Missed requirement or edge case, if any.\n"
    "3. Concrete change: what the next attempt must do differently.\n"
    "4. What NOT to repeat from earlier attempts.\n"
    "Be specific and terse. If the root cause is genuinely unclear, say so."
)


def reflect_on_failure(
    task_description: Optional[str],
    error_output: Optional[str],
    prior_reflections: Optional[List[str]] = None,
    reflect_call: Optional[ReflectCall] = None,
    llm_client=None,
    reflect_model: Optional[str] = None,
    timeout: float = 45,
    generator_model: Optional[str] = None,
) -> str:
    """Produce a short reflection on why an attempt failed, or "" to skip.

    The model call is performed by ``reflect_call`` when supplied (the test
    seam); otherwise one is built from ``llm_client``. ``reflect_model`` selects
    an independent model when given. ``prior_reflections`` are earlier attempts'
    reflections, folded in so the model builds on (rather than repeats) them.
    ``generator_model`` is the actual model that produced the failed attempt
    (when the caller knows it), so the same-model independence check compares
    against the real generator rather than the client's static default.

    Returns a trimmed reflection string, or "" on no client, empty output, any
    error, or the hard timeout — so a failed reflection never blocks the retry.
    """
    if not (error_output and error_output.strip()):
        return ""
    call = reflect_call or _default_reflect_call(
        llm_client, reflect_model, generator_model=generator_model
    )
    if call is None:
        return ""

    prior = ""
    if prior_reflections:
        joined = "\n".join(f"- {r}" for r in prior_reflections if r.strip())
        if joined:
            prior = f"\n## Your earlier reflections (do not repeat these)\n{joined}\n"
    prompt = _PROMPT.format(
        task=(task_description or "(none given)").strip(),
        output=error_output.strip()[-_MAX_OUTPUT_CHARS:],
        prior=prior,
    )

    def _work() -> str:
        try:
            text = call(prompt) or ""
        except Exception as e:  # any model/IO failure is non-fatal -> skip
            logger.debug(f"Reflection unavailable: {e}")
            return ""
        return text.strip()[:_MAX_REFLECTION_CHARS]

    return run_bounded(_work, timeout, "", "Failure reflection")


def _default_reflect_call(
    llm_client, reflect_model: Optional[str], *, generator_model: Optional[str] = None
) -> Optional[ReflectCall]:
    """Build a reflection call from the project's LLM client, or None if unusable."""
    system = (
        "You are a precise debugging assistant reflecting on a failed coding "
        "attempt. Output only the requested short reflection."
    )
    return build_independent_call(
        llm_client,
        system,
        reflect_model,
        "Reflection",
        generator_model=generator_model,
    )
