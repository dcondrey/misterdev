"""The mutation proposer — turn blame into a concrete, targeted self-edit.

This is a thin adapter over misterdev's OWN edit machinery (anchored
SEARCH/REPLACE + context assembly), which the design names as the mutation
operator — the proposer does not reinvent code generation. Its real job is the
*targeting*: build an instruction that points the editor at the highest-blame
niche, seeded with real failure examples and biased toward the mutation kinds the
archive has shown pay off (the prior), then normalize the editor's response into
a :class:`Mutation` (files touched + patch + kind tag) the loop can guardrail,
sandbox, and archive.

The editor call is INJECTED (``generate(instruction) -> response``) so this stays
pure and testable; the driver wires it to ``project.llm_client`` + the real edit
pipeline, which supplies the source context anchored edits need.
"""

import re
from typing import Callable, List, Optional

from .attribution import Blame
from .loop import Mutation

# Matches an anchored-edit fence line ``` ```<lang>:<path> ``` — the path is what
# the mutation actually touches (see EDIT_FORMAT_INSTRUCTIONS).
_FENCE_PATH_RE = re.compile(r"^`{3,}[\w+.\-]*:(\S+)\s*$", re.MULTILINE)
# An optional leading ``tag: <kind>`` line the proposer asks for, so the archive
# can learn which KINDS of edit pay off (feeds MutationPrior).
_TAG_RE = re.compile(r"^\s*tag:\s*([\w./-]+)", re.IGNORECASE | re.MULTILINE)


def parse_paths(response: str) -> List[str]:
    """The distinct file paths an edit response touches, in first-seen order."""
    seen: List[str] = []
    for m in _FENCE_PATH_RE.finditer(response or ""):
        p = m.group(1)
        if p not in seen:
            seen.append(p)
    return seen


def parse_tag(response: str) -> Optional[str]:
    """The proposer's self-declared mutation kind, if it emitted a ``tag:`` line."""
    m = _TAG_RE.search(response or "")
    return m.group(1).lower() if m else None


def build_instruction(blame: Blame, favored_kinds: Optional[List[str]] = None) -> str:
    """The targeting instruction handed to the editor.

    Focuses the edit on the blamed niche, shows real failures to fix, biases
    toward proven mutation kinds when the prior has evidence, and asks for a
    ``tag:`` line so the outcome can teach the prior. It does NOT restate the
    edit-format rules — the editor pipeline already supplies those.
    """
    lines = [
        f"## Self-improvement target: {blame.niche}",
        (
            f"misterdev fails {blame.failures}/{blame.total} "
            f"({blame.failure_rate:.0%}) of benchmark tasks in this niche. Propose "
            "the SMALLEST edit to misterdev's own source that would make these fail "
            "cases pass, without regressing anything else."
        ),
    ]
    if favored_kinds:
        lines.append(
            "Prior runs show these edit kinds pay off here; prefer them if apt: "
            + ", ".join(favored_kinds)
            + "."
        )
    if blame.examples:
        lines.append("\n### Representative failures")
        for i, ex in enumerate(blame.examples, 1):
            lines.append(f"Failure {i}:\n```\n{ex}\n```")
    lines.append(
        "\nBegin your reply with a single line `tag: <kind>` naming the kind of "
        "change (e.g. `tag: contract-extraction`, `tag: prompt`, `tag: gate-tuning`)."
    )
    return "\n".join(lines)


class LLMProposer:
    """Blame -> targeted :class:`Mutation`, via an injected editor call."""

    def __init__(self, generate: Callable[[str], str]):
        self.generate = generate

    def propose(
        self, blame: Blame, favored_kinds: Optional[List[str]] = None
    ) -> Mutation:
        """Produce a targeted mutation for ``blame``.

        Raises ``ValueError`` when the editor returns nothing editable (no parseable
        file path), so the loop counts it as a dead candidate rather than sandboxing
        an empty diff. The kind tag defaults to the niche when the editor omits one.
        """
        response = self.generate(build_instruction(blame, favored_kinds))
        paths = parse_paths(response)
        if not paths:
            raise ValueError("proposal contained no editable file")
        return Mutation(
            target=blame.niche,
            paths=paths,
            patch=response,
            note=parse_tag(response) or blame.niche,
        )
