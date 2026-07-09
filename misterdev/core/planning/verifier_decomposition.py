"""Verifier decomposition: synthesize dense intermediate checks from a contract.

WHY (docs/path-to-100.md, section D3): a hard task with ONE final test has a
sparse reward gradient. The model's per-attempt success probability `p` on a
state-heavy goal (a type, its mutators, and its derivations, all landing at once)
is tiny, and you cannot out-search a tiny `p`. Densify the reward instead: split
the goal into ordered, independently-verifiable STAGES synthesized from the task's
own public-API contract (state before operations before derivations). Each stage
is a subgoal with its own smoke-check, so a single low-`p` verified goal becomes a
chain of high-`p` verified subgoals and the search converges.

This module is PURE, deterministic ordering logic over an extracted contract's
public symbols. No LLM, no I/O, no wall-clock. A decomposer feeds
`render_stages(synthesize_stages(symbols))` into its decomposition prompt so a
complex single file is staged into 2-3 dependency-ordered sub-tasks.

Symbols are duck-typed: each exposes a `.name` and optionally a `.kind`. The
extractor emits dicts ({name, kind, signature}); this module accepts either
attribute or mapping access so it works on both without coupling to a concrete
type.
"""

from dataclasses import dataclass

# Construction: the type/state and its entry points. Matched exactly (case-
# insensitive) so a query like "getitem" or a mutator "resetter" is not mistaken
# for a constructor by a naive substring test.
_CONSTRUCTOR_NAMES = frozenset(
    {"new", "init", "__init__", "default", "create", "build", "make", "from"}
)

# Mutators/commands: state transitions. Matched as a leading verb (prefix) so
# "add", "add_frame", "set_value", "push_back" all land in stage 2.
_MUTATOR_PREFIXES = (
    "add",
    "set",
    "push",
    "pop",
    "roll",
    "insert",
    "remove",
    "delete",
    "update",
    "append",
    "put",
    "write",
    "clear",
    "reset",
    "apply",
    "move",
    "toggle",
)

# Queries/derivations: read-only reads over the state built in stages 1-2.
_QUERY_PREFIXES = (
    "score",
    "get",
    "value",
    "result",
    "is_",
    "has_",
    "count",
    "total",
    "sum",
    "len",
    "size",
    "read",
    "find",
    "compute",
    "calculate",
    "peek",
    "to_",
    "as_",
)

_STAGE_CONSTRUCTOR = 1
_STAGE_MUTATOR = 2
_STAGE_QUERY = 3

# Symbols that match no bucket default here: they are read-shaped by nature (a
# free helper, an accessor) and safest verified last, after state exists.
_STAGE_DEFAULT = _STAGE_QUERY


@dataclass
class StagedCheck:
    """One ordered, independently-verifiable stage of a decomposed goal.

    `stage` is the 1-based execution order (construction -> mutation -> query).
    `symbol` is the public symbol name the stage implements and smoke-checks.
    `description` is the one-line "done" condition a sub-task carries.
    `rationale` says WHY the stage sits where it does in the dependency order.
    """

    stage: int
    symbol: str
    description: str
    rationale: str


def _symbol_name(symbol) -> str:
    """Read `.name` from an attribute object or a {name: ...} mapping."""
    if isinstance(symbol, dict):
        name = symbol.get("name", "")
    else:
        name = getattr(symbol, "name", "")
    return (name or "").strip()


def _symbol_kind(symbol) -> str:
    """Read `.kind` (optional) from an attribute object or mapping. May be ''."""
    if isinstance(symbol, dict):
        kind = symbol.get("kind", "")
    else:
        kind = getattr(symbol, "kind", "")
    return (kind or "").strip().lower()


def _classify(name: str, kind: str) -> int:
    """Map a symbol to its stage (1 construction, 2 mutation, 3 query).

    Deterministic and precedence-ordered: kind hints first, then name shape.
    `is_*`/`has_*` are checked before the generic verb-prefix scan so a boolean
    query is never captured by a mutator prefix.
    """
    lname = name.lower()

    # Explicit kind hints from the extractor take priority over name shape: a
    # type/class/struct/enum is the state itself and belongs in stage 1.
    if kind in ("class", "struct", "enum", "type", "trait", "interface", "protocol"):
        return _STAGE_CONSTRUCTOR
    if kind in ("constructor", "init", "initializer"):
        return _STAGE_CONSTRUCTOR

    # Constructor by name (exact match, case-insensitive).
    if lname in _CONSTRUCTOR_NAMES:
        return _STAGE_CONSTRUCTOR

    # Query booleans first so "is_*"/"has_*" win over any verb-prefix overlap.
    if lname.startswith(_QUERY_PREFIXES):
        return _STAGE_QUERY

    # Mutators/commands.
    if lname.startswith(_MUTATOR_PREFIXES):
        return _STAGE_MUTATOR

    return _STAGE_DEFAULT


def _stage_rationale(stage: int) -> str:
    """The dependency-order justification injected into each stage's sub-task."""
    if stage == _STAGE_CONSTRUCTOR:
        return (
            "State must exist before anything can act on it: build the type and "
            "its constructor first so later stages have something to verify against."
        )
    if stage == _STAGE_MUTATOR:
        return (
            "Mutators drive state transitions and depend on the constructor from "
            "stage 1; verifying them in isolation gives a dense mid-goal reward."
        )
    return (
        "Queries/derivations read the state produced by stages 1-2; verifying "
        "them last confirms the whole pipeline without a single sparse final test."
    )


def synthesize_stages(symbols, instructions: str = "") -> list[StagedCheck]:
    """Turn a contract's public symbols into an ordered staged plan.

    Ordering heuristic (pure, no LLM): construction (stage 1) -> mutators
    (stage 2) -> queries/derivations (stage 3). Within a stage, symbols keep
    their contract order so the output is fully deterministic. Empty stages are
    dropped and the surviving stages are renumbered to be contiguous (a
    query-only contract yields a single stage numbered 1).

    `instructions` is accepted for interface parity with the decomposer (it may
    later bias classification) but does not affect ordering today, keeping this
    a pure function of the contract.

    Returns [] for an empty or None contract.
    """
    if not symbols:
        return []

    del instructions  # reserved; ordering is a pure function of the contract

    # Bucket by stage while preserving contract order within each bucket.
    buckets: dict[int, list[str]] = {
        _STAGE_CONSTRUCTOR: [],
        _STAGE_MUTATOR: [],
        _STAGE_QUERY: [],
    }
    seen: set[str] = set()
    for symbol in symbols:
        name = _symbol_name(symbol)
        if not name or name in seen:
            continue
        seen.add(name)
        buckets[_classify(name, _symbol_kind(symbol))].append(name)

    # Renumber non-empty buckets to contiguous 1..N so a mutator-only or
    # query-only contract still starts at stage 1.
    checks: list[StagedCheck] = []
    next_stage = 1
    for logical_stage in (_STAGE_CONSTRUCTOR, _STAGE_MUTATOR, _STAGE_QUERY):
        names = buckets[logical_stage]
        if not names:
            continue
        rationale = _stage_rationale(logical_stage)
        for name in names:
            checks.append(
                StagedCheck(
                    stage=next_stage,
                    symbol=name,
                    description=f"implement + smoke-check {name}",
                    rationale=rationale,
                )
            )
        next_stage += 1

    return checks


def render_stages(stages: list[StagedCheck]) -> str:
    """Render staged checks as a compact text block for a decomposition prompt.

    Groups symbols by stage number so a decomposer reads one line per stage
    ("Stage 1: ... / Stage 2: ..."), suitable to inject verbatim into a prompt.
    Returns "" for an empty plan.
    """
    if not stages:
        return ""

    # Preserve first-seen stage order (stages arrive already ordered).
    order: list[int] = []
    by_stage: dict[int, list[StagedCheck]] = {}
    for check in stages:
        if check.stage not in by_stage:
            by_stage[check.stage] = []
            order.append(check.stage)
        by_stage[check.stage].append(check)

    lines: list[str] = []
    for stage in order:
        members = by_stage[stage]
        symbols = ", ".join(c.symbol for c in members)
        rationale = members[0].rationale
        lines.append(f"Stage {stage}: {symbols} — {rationale}")
    return "\n".join(lines)
