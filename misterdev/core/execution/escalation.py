"""Escalation ladder for a task that keeps failing the same way.

A task retried identically tends to fail identically. This ladder escalates the
RESPONSE as a task accumulates NON-INFRA (real code) failures, in order:

    normal -> widen_context -> full_rewrite -> stronger_model -> decompose

- ``widen_context``: show the full target file(s), their dependents, and the
  task's pre-decided spec/acceptance verbatim — the model was probably editing
  blind or against a truncated view.
- ``full_rewrite``: a STRUCTURALLY different attempt — stop patching (SEARCH/
  REPLACE against a view the model keeps mis-anchoring on) and rewrite the whole
  target region from scratch. This is not another reprompt of the same approach;
  it changes HOW the edit is produced while keeping the same task and model.
- ``stronger_model``: the cheap/default model can't crack it; route the
  generation to a more capable one.
- ``decompose``: the task is too big to land in one edit; request a split into
  named sub-steps instead of failing outright.

Crucially, an ENVIRONMENT fault (a timeout, a locked store — see ``infra.py``)
self-heals elsewhere and is NOT the code's fault, so it must never advance the
ladder. ``should_count_failure`` is the gate: only failures it counts move the
rung. The rung choice itself is a pure function of that count, so the whole
policy is unit-testable in isolation.
"""

from misterdev.core.execution.blocker import blocked_reason
from misterdev.core.execution.infra import infra_failure

# In escalating order; index doubles as the rung's strength.
RUNGS = ("normal", "widen_context", "full_rewrite", "stronger_model", "decompose")


def should_count_failure(output: str) -> bool:
    """True when a gate failure is a real CODE failure that advances the ladder.

    An environment/infra fault (timeout, locked store, OOM) or a blocked request
    (auth failure, quota exhausted) self-heals or is unretriable, so it returns
    False and the ladder holds where it is.
    """
    return infra_failure(output) is None and blocked_reason(output) is None


def choose_rung(
    code_failures: int,
    *,
    widen_after: int = 1,
    rewrite_after: int = 2,
    model_after: int = 3,
    decompose_after: int = 3,
) -> str:
    """The escalation rung for the NEXT attempt, given prior NON-infra failures.

    ``code_failures`` is the number of real code failures the task has already
    taken (0 on the first attempt -> ``normal``). Highest threshold that is met
    wins, so misordered thresholds still yield a defined (monotonic) rung. The
    default ``model_after``/``decompose_after`` coincide at 3 so that within the
    default 3-attempt budget a task climbs normal -> widen -> full_rewrite and then
    decomposes at exhaustion; ``stronger_model`` (a no-op unless ``escalation_model``
    is set) becomes reachable only when ``decompose_after`` is raised.
    """
    if code_failures >= decompose_after:
        return "decompose"
    if code_failures >= model_after:
        return "stronger_model"
    if code_failures >= rewrite_after:
        return "full_rewrite"
    if code_failures >= widen_after:
        return "widen_context"
    return "normal"
