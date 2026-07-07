"""Rule model + relevance selection for injected best-practice guidance.

Guidance is authored as a list of :class:`Rule`s instead of one prose block, so
only the rules that pertain to a given edit are injected. A rule is either
``core`` (a small always-on baseline) or gated behind ``triggers`` — lowercase
substrings that, when any appears in the task context (description, acceptance
criteria, error logs, code), make the rule relevant. Rule text is terse and
symbolic (`→` then/implies, `/` or, `>` prefer-over) to hold token cost down
without a decode legend.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    """One best-practice guideline. ``core`` rules are always emitted; the rest
    are emitted only when a trigger substring is present in the context."""

    text: str
    triggers: tuple = ()
    core: bool = False


def select_rules(rules, context: str, *, max_rules: int = 11, max_chars: int = 1500):
    """Return core rules plus any whose triggers match ``context``, capped.

    Core rules are kept even past the caps (they are the baseline); triggered
    rules fill the remaining budget in declaration order.
    """
    ctx = (context or "").lower()
    core = [r for r in rules if r.core]
    matched = [r for r in rules if not r.core and any(t in ctx for t in r.triggers)]

    out = list(core)
    used = sum(len(r.text) for r in out)
    for r in matched:
        if len(out) >= max_rules or used + len(r.text) > max_chars:
            break
        out.append(r)
        used += len(r.text)
    return out


def render_rules(title: str, rules) -> str:
    """Render selected rules as a titled bullet list, or "" when none."""
    if not rules:
        return ""
    lines = [f"{title} — apply the relevant best-practice rules:"]
    lines.extend(f"- {r.text}" for r in rules)
    return "\n".join(lines)
