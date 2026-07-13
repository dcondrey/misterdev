"""Localization: given a goal/issue and no target files, FIND the ones to edit.

misterdev's pipeline assumes decomposition already knows a task's target files;
when it doesn't (a bare issue, "fix the X bug"), every downstream stage pays for
the mis-scope (docs/research-directions.md, Theme 4). This is the missing
navigator: it ranks the existing tree-sitter symbol graph against the query to
surface the most relevant functions/methods and the files that own them —
function-level targets before editing, on the parse misterdev already built (no
new index, no embedding service required).

Signals, cheap and multilingual:
  - term overlap between the query and each symbol's NAME (split on camelCase /
    snake_case so "user name" matches ``getUserName``), weighted highest;
  - an exact/whole-name hit boost;
  - overlap with the owning file's basename;
  - whole-word query hits inside the symbol body (identifiers / docstring);
  - a decayed bonus propagated across CALL EDGES, so a strong hit pulls in its
    callers and callees (the fix often lives one hop from the named symbol).

Optional ``ranker`` (the project's :class:`SemanticRanker`) does two jobs: it
selects the best ``top_k`` from the lexical over-fetch, and — when lexical
scoring is empty or weak (a vocabulary mismatch, e.g. the issue says "logins are
slow" while the code says ``authenticate``) — it ESCALATES to a semantic search
over the whole symbol graph, reaching targets pure lexical never could. The
ranker is duck-typed on ``.top_k(query, {id: text}, k) -> [id]`` (the interface
used everywhere else in the project); it is best-effort, so any failure degrades
to the lexical order and this module never raises.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z0-9]+|[A-Z]+|[0-9]+")
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "to",
        "of",
        "in",
        "on",
        "for",
        "and",
        "or",
        "is",
        "it",
        "fix",
        "bug",
        "add",
        "make",
        "when",
        "with",
        "that",
        "this",
        "should",
        "error",
        "issue",
        "problem",
        "code",
        "function",
        "method",
        "class",
    }
)

_NAME_WEIGHT = 3.0
_EXACT_NAME_BONUS = 5.0
_PATH_WEIGHT = 1.0
_BODY_WEIGHT = 0.4
_BODY_CAP = 3.0
_EDGE_DECAY = 0.35  # bonus a hit propagates to each call-graph neighbor
# Below this top lexical score no query term matched a symbol NAME (only weaker
# path/body/expansion signal), so the lexical result is untrustworthy and the
# semantic fallback is worth its (cached) cost.
_WEAK_LEXICAL_FLOOR = _NAME_WEIGHT
# Hard bound on how many symbols the semantic fallback embeds in one call, so a
# huge graph can't turn a mis-scope into an unbounded embedding bill. Truncation
# is logged, never silent.
_FALLBACK_POOL_CAP = 400
_SNIPPET_CHARS = 500


def _split(text: str) -> set:
    """Tokens from text, splitting identifiers on camelCase and snake/kebab."""
    out = set()
    for word in _WORD_RE.findall(text or ""):
        out.add(word.lower())
        for piece in _CAMEL_RE.findall(word):
            if piece:
                out.add(piece.lower())
    return out


def _query_terms(query: str) -> set:
    return {t for t in _split(query) if t not in _STOPWORDS and len(t) > 1}


@dataclass(frozen=True)
class LocalizationHit:
    file_path: str
    symbol: str
    kind: str
    score: float


def _score_symbol(sym: Any, terms: set) -> float:
    name_tokens = _split(getattr(sym, "name", "") or "")
    score = _NAME_WEIGHT * len(terms & name_tokens)
    if (getattr(sym, "name", "") or "").lower() in terms:
        score += _EXACT_NAME_BONUS
    base = (getattr(sym, "file_path", "") or "").rsplit("/", 1)[-1]
    score += _PATH_WEIGHT * len(terms & _split(base))
    body_tokens = _split(getattr(sym, "content", "") or "")
    score += min(_BODY_WEIGHT * len(terms & body_tokens), _BODY_CAP)
    return score


def _snippet(sym: Any) -> str:
    """Short representative text for a symbol, for the semantic ranker."""
    return (getattr(sym, "content", "") or getattr(sym, "name", "") or "")[
        :_SNIPPET_CHARS
    ]


def _hit(sym: Any, score: float) -> LocalizationHit:
    return LocalizationHit(
        file_path=getattr(sym, "file_path", "") or "",
        symbol=getattr(sym, "name", "") or "",
        kind=getattr(sym, "kind", "") or "",
        score=round(score, 3),
    )


def _semantic_fallback(
    query: str,
    symbols: Dict[str, Any],
    top_k: int,
    ranker: Any,
    scored: Dict[str, float],
) -> List[LocalizationHit]:
    """Escalate a weak/empty lexical result to a semantic search over the whole
    graph. Returns semantically ranked hits, or ``[]`` if the ranker yields
    nothing (so the caller can fall back to whatever lexical found).

    ``top_k`` engages ranking only when it selects a strict subset (its contract),
    so for pools at or below ``top_k`` the ranker returns everything unranked; we
    pre-order the pool by the weak lexical score first, giving a sensible,
    deterministic order in that case too."""
    ordered = sorted(symbols, key=lambda k: (-scored.get(k, 0.0), k))
    if len(ordered) > _FALLBACK_POOL_CAP:
        logger.info(
            "localize: semantic fallback pool truncated to %d of %d symbols",
            _FALLBACK_POOL_CAP,
            len(ordered),
        )
        ordered = ordered[:_FALLBACK_POOL_CAP]
    candidates = {k: _snippet(symbols[k]) for k in ordered}
    try:
        chosen = ranker.top_k(query, candidates, top_k)
    except Exception:  # semantic retrieval is best-effort; never raise
        return []
    n = len(chosen)
    # Synthetic descending score in (0, 1]: preserves the ranker's order and lets
    # localize_files() sum per file, without pretending to be a lexical score.
    return [_hit(symbols[k], (n - i) / n) for i, k in enumerate(chosen) if k in symbols]


def localize(
    query: str,
    symbols: Dict[str, Any],
    *,
    top_k: int = 10,
    expand: bool = True,
    ranker: Optional[Any] = None,
) -> List[LocalizationHit]:
    """Rank symbols by relevance to ``query``; return the top ``top_k`` hits.

    ``expand`` propagates a decayed bonus across call edges so a strong hit's
    neighbors surface too. ``ranker`` (optional, a :class:`SemanticRanker`) both
    selects the best ``top_k`` from the lexical over-fetch AND rescues a weak or
    empty lexical result by searching the whole graph semantically; any failure
    falls back to the lexical order.
    """
    terms = _query_terms(query)
    if not terms or not symbols:
        return []
    scored: Dict[str, float] = {}
    for key, sym in symbols.items():
        s = _score_symbol(sym, terms)
        if s > 0:
            scored[key] = s

    # Vocabulary mismatch: lexical found nothing confident (no query term hit a
    # symbol NAME). Escalate to semantic retrieval over the whole graph, which
    # reaches targets pure lexical can't. Only when a ranker is available.
    lexical_is_weak = not scored or max(scored.values()) < _WEAK_LEXICAL_FLOOR
    if lexical_is_weak and ranker is not None:
        rescued = _semantic_fallback(query, symbols, top_k, ranker, scored)
        if rescued:
            return rescued

    if expand and scored:
        bonus: Dict[str, float] = {}
        by_name: Dict[str, List[str]] = {}
        for key, sym in symbols.items():
            by_name.setdefault(getattr(sym, "name", "") or "", []).append(key)
        for key, base_score in scored.items():
            sym = symbols[key]
            neighbors = set(getattr(sym, "outgoing_calls", set())) | set(
                getattr(sym, "incoming_calls", set())
            )
            for nb in neighbors:
                for nb_key in by_name.get(nb, [nb]):
                    if nb_key in symbols:
                        bonus[nb_key] = (
                            bonus.get(nb_key, 0.0) + base_score * _EDGE_DECAY
                        )
        for key, b in bonus.items():
            scored[key] = scored.get(key, 0.0) + b

    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    shortlist = ranked[: max(top_k * 3, top_k)]
    # Semantic rerank: pick the best top_k from the lexical over-fetch. top_k only
    # reorders when selecting a strict subset (its contract), so gate on the
    # shortlist genuinely exceeding top_k — otherwise the lexical order stands.
    if ranker is not None and len(shortlist) > top_k:
        try:
            score_map = dict(shortlist)
            candidates = {k: _snippet(symbols[k]) for k in score_map}
            chosen = ranker.top_k(query, candidates, top_k)
            if chosen:
                shortlist = [(k, score_map[k]) for k in chosen if k in score_map]
        except Exception:  # semantic rerank is best-effort; keep lexical order
            pass

    return [_hit(symbols[key], score) for key, score in shortlist[:top_k]]


def localize_files(
    query: str, symbols: Dict[str, Any], *, top_k: int = 5, **kw
) -> List[Tuple[str, float]]:
    """File-level targets: sum symbol relevance per file, best-first. This is the
    shape decomposition wants — which files to scope a task to."""
    per_file: Dict[str, float] = {}
    for hit in localize(query, symbols, top_k=10_000, **kw):
        per_file[hit.file_path] = per_file.get(hit.file_path, 0.0) + hit.score
    return sorted(per_file.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
