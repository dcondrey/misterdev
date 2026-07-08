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
from pathlib import Path
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


# misterdev's real editable surfaces, so the editor proposes edits at paths that
# EXIST and are wired in — not plausible-looking invented files (observed: the
# unguided editor proposed `src/prompts/...`, which misterdev has no concept of,
# yielding an inert mutation). Also steers toward a GENERAL mechanism that removes
# a whole failure class over a niche-specific tweak (which would overfit).
_STRUCTURAL_SURFACES = (
    "misterdev's editable structural surfaces — edit one of THESE (or add a module "
    "beside it); do NOT invent paths:\n"
    "- misterdev/core/context/guidance/<lang>.py — per-language best-practice RULES, "
    "relevance-selected. Add/adjust a GENERAL rule, never a task-specific one.\n"
    "- misterdev/task_executors/markdown_plan_executor/gates_mixin.py — correctness "
    "gates and the error context fed back to the model on a failure.\n"
    "- misterdev/task_executors/markdown_plan_executor/edits_mixin.py — edit "
    "application and the structural guards (dangling-ref, test-tamper).\n"
    "- misterdev/core/execution/failure_view.py — parses runner output into exact "
    "expected/actual (the observation seam).\n"
    "- misterdev/core/planning/decomposer.py — how a goal is split into tasks.\n"
    "- misterdev/core/execution/error_classifier.py — how errors are classified.\n"
)

# The kind of fix that fits each L2 failure cause — so the editor aims at the
# mechanism, not a symptom. (saturation is a capability signal, not a fix target.)
_CAUSE_FIX = {
    "artifact": "fix the guard/gate that wrongly blocked a correct edit "
    "(edits_mixin.py / gates_mixin.py) — a correct solution must never be rejected",
    "observation": "improve the observation seam so the model sees the exact "
    "failure (failure_view.py / error_classifier.py)",
    "convergence": "force approach diversity on repeated failure so the model "
    "stops thrashing one strategy (the retry/decomposition path)",
    "search": "make the search cheaper or better-guided so it converges within "
    "budget (gates_mixin.py error context / decomposer.py)",
    "saturation": "this looks like a capability wall, not a harness defect — a "
    "guidance rule that raises solution quality is the only apt edit; do not hack",
}


def build_instruction(blame: Blame, favored_kinds: Optional[List[str]] = None) -> str:
    """The targeting instruction handed to the editor.

    Focuses the edit on the blamed niche, shows real failures to fix, grounds the
    editor in misterdev's real editable surfaces, steers toward a GENERAL fix over
    a niche tweak, biases toward proven mutation kinds when the prior has evidence,
    and asks for a ``tag:`` line so the outcome can teach the prior. It does NOT
    restate the edit-format rules — the editor pipeline already supplies those.
    """
    lines = [
        f"## Self-improvement target: {blame.niche}",
        (
            f"misterdev fails {blame.failures}/{blame.total} "
            f"({blame.failure_rate:.0%}) of {blame.source} in this niche. Propose an "
            "edit to misterdev's own source that makes these cases pass WITHOUT "
            "regressing anything else. Prefer a GENERAL mechanism — a guard, a "
            "parser/observation seam, a guidance rule, a gate — that removes this "
            "whole class of failure, NOT a change keyed to these specific tasks "
            "(that would overfit and is rejected)."
        ),
        "\n" + _STRUCTURAL_SURFACES,
    ]
    cause = getattr(blame, "cause", "") or ""
    if cause:
        matching = _CAUSE_FIX.get(cause, "a structural fix that removes this class")
        lines.append(
            f"\nFailure cause (classified): {cause}"
            + (
                f" — {blame.cause_evidence}"
                if getattr(blame, "cause_evidence", "")
                else ""
            )
            + f". The fitting fix: {matching}"
        )
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
        "change (e.g. `tag: guard`, `tag: guidance-rule`, `tag: observation-seam`, "
        "`tag: gate-tuning`, `tag: contract-extraction`)."
    )
    return "\n".join(lines)


class LLMProposer:
    """Blame -> targeted :class:`Mutation`, via an injected editor call."""

    def __init__(
        self, generate: Callable[[str], str], repo_root: Optional[object] = None
    ):
        self.generate = generate
        self.repo_root = Path(repo_root) if repo_root is not None else None

    def _grounded(self, path: str) -> bool:
        """A path is real when it exists, or is a new file in an existing dir.

        Rejects invented paths (a nonexistent parent dir), so an inert mutation
        that edits files misterdev has no concept of never reaches the paid
        sandbox. With no repo_root, validation is skipped (kept for pure tests).
        """
        if self.repo_root is None:
            return True
        target = self.repo_root / path
        return target.exists() or target.parent.is_dir()

    def propose(
        self, blame: Blame, favored_kinds: Optional[List[str]] = None
    ) -> Mutation:
        """Produce a targeted mutation for ``blame``.

        Raises ``ValueError`` when the editor returns nothing editable (no parseable
        file path) or every path it names is invented (no such directory) — so the
        loop counts it a dead candidate rather than sandboxing an inert diff. The
        kind tag defaults to the niche when the editor omits one.
        """
        response = self.generate(build_instruction(blame, favored_kinds))
        paths = parse_paths(response)
        if not paths:
            raise ValueError("proposal contained no editable file")
        grounded = [p for p in paths if self._grounded(p)]
        if not grounded:
            raise ValueError(
                f"proposal names only invented paths (not in repo): {paths}"
            )
        return Mutation(
            target=blame.niche,
            paths=grounded,
            patch=response,
            note=parse_tag(response) or blame.niche,
        )
