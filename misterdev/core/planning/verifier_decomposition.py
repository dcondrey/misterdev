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


def _edge_keys(symbol, field: str) -> frozenset[str]:
    """Read a call-graph edge set (`incoming_calls`/`outgoing_calls`) off a symbol.

    Duck-typed: accepts attribute objects or {field: ...} mappings, and any
    iterable of caller/callee keys (set, list, tuple). Every key is coerced to a
    stripped string so heterogeneous edge payloads still layer deterministically.
    Never raises: a missing field, a non-iterable value, or an un-stringable
    element yields the empty set, which the caller treats as "no edge info".
    """
    if isinstance(symbol, dict):
        raw = symbol.get(field)
    else:
        raw = getattr(symbol, field, None)
    if not raw or isinstance(raw, (str, bytes)):
        # A bare string is a scalar, not an edge set; reject it rather than
        # iterating it character by character.
        return frozenset()
    try:
        keys = {str(k).strip() for k in raw}
    except TypeError:
        return frozenset()
    return frozenset(k for k in keys if k)


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


def _dependency_rationale() -> str:
    """The justification injected into a stage ordered by call-graph depth."""
    return (
        "Call-graph depth orders this stage: a symbol called by others is a "
        "dependency and is built first, so its callers have a verified "
        "foundation to lean on before their own smoke-check runs."
    )


def _edge_layers(
    ordered_names: list[str], edges: dict[str, tuple[frozenset, frozenset]]
):
    """Layer symbols by call-graph depth: dependencies (called-by-others) first.

    Each symbol's depth is (# outgoing edges to other in-contract symbols) minus
    (# incoming edges from them); a foundational symbol that many callers depend
    on and that calls few others scores lowest and lands in an earlier stage.
    Only intra-contract edges count so an external callee never skews a layer.
    Ties keep contract order, so the result is fully deterministic.

    Returns a list of (depth, name) buckets grouped into contiguous stages, or
    None when the resulting layering is degenerate (a single layer), in which
    case the caller keeps the name-verb heuristic.
    """
    in_contract = set(ordered_names)
    scored: list[tuple[int, int, str]] = []
    for index, name in enumerate(ordered_names):
        incoming, outgoing = edges.get(name, (frozenset(), frozenset()))
        depth = len(outgoing & in_contract) - len(incoming & in_contract)
        scored.append((depth, index, name))

    distinct_depths = sorted({depth for depth, _, _ in scored})
    if len(distinct_depths) < 2:
        return None

    depth_to_stage = {
        depth: stage for stage, depth in enumerate(distinct_depths, start=1)
    }
    ordered = sorted(scored, key=lambda item: (depth_to_stage[item[0]], item[1]))
    return [(depth_to_stage[depth], name) for depth, _, name in ordered]


def synthesize_stages(symbols, instructions: str = "") -> list[StagedCheck]:
    """Turn a contract's public symbols into an ordered staged plan.

    When the symbols carry call-graph edges (`incoming_calls`/`outgoing_calls`,
    duck-typed) the plan is ordered by dependency depth: a symbol called by
    others is foundational and is staged before its callers, giving a truer
    build order than name shape alone. When NO symbol exposes edge info (or the
    edges collapse to a single layer) the plan falls back to the name-verb
    heuristic below: construction (stage 1) -> mutators (stage 2) ->
    queries/derivations (stage 3). Malformed edge payloads never raise; they are
    read as "no edge info" and trigger the same fallback.

    Within a stage, symbols keep their contract order so the output is fully
    deterministic. Empty stages are dropped and the surviving stages are
    renumbered to be contiguous (a query-only contract yields a single stage
    numbered 1).

    `instructions` is accepted for interface parity with the decomposer (it may
    later bias classification) but does not affect ordering today, keeping this
    a pure function of the contract.

    Returns [] for an empty or None contract.
    """
    if not symbols:
        return []

    del instructions  # reserved; ordering is a pure function of the contract

    # First pass: dedup, preserve contract order, capture kind, and collect any
    # edge payloads so both ordering paths read the contract exactly once.
    ordered_names: list[str] = []
    kinds: dict[str, str] = {}
    edges: dict[str, tuple[frozenset, frozenset]] = {}
    has_edges = False
    seen: set[str] = set()
    for symbol in symbols:
        name = _symbol_name(symbol)
        if not name or name in seen:
            continue
        seen.add(name)
        ordered_names.append(name)
        kinds[name] = _symbol_kind(symbol)
        incoming = _edge_keys(symbol, "incoming_calls")
        outgoing = _edge_keys(symbol, "outgoing_calls")
        edges[name] = (incoming, outgoing)
        if incoming or outgoing:
            has_edges = True

    if not ordered_names:
        return []

    # Edge-driven ordering when the contract carries a usable call graph.
    if has_edges:
        layered = _edge_layers(ordered_names, edges)
        if layered is not None:
            rationale = _dependency_rationale()
            return [
                StagedCheck(
                    stage=stage,
                    symbol=name,
                    description=f"implement + smoke-check {name}",
                    rationale=rationale,
                )
                for stage, name in layered
            ]

    # Fallback: name-verb heuristic. Bucket by stage, preserving contract order.
    buckets: dict[int, list[str]] = {
        _STAGE_CONSTRUCTOR: [],
        _STAGE_MUTATOR: [],
        _STAGE_QUERY: [],
    }
    for name in ordered_names:
        buckets[_classify(name, kinds[name])].append(name)

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
