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

Optional ``ranker`` (the project's semantic ranker) re-scores the lexical
shortlist. Pure over a ``{key: SymbolNode}`` mapping — no I/O — so it is fully
testable and never raises.
"""

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

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


def localize(
    query: str,
    symbols: Dict[str, Any],
    *,
    top_k: int = 10,
    expand: bool = True,
    ranker: Optional[Callable[[str, List[str]], List[int]]] = None,
) -> List[LocalizationHit]:
    """Rank symbols by relevance to ``query``; return the top ``top_k`` hits.

    ``expand`` propagates a decayed bonus across call edges so a strong hit's
    neighbors surface too. ``ranker`` (optional) re-orders the lexical shortlist
    by semantic similarity: it takes ``(query, [snippet, ...])`` and returns the
    indices best-first; any failure falls back to the lexical order.
    """
    terms = _query_terms(query)
    if not terms or not symbols:
        return []
    scored: Dict[str, float] = {}
    for key, sym in symbols.items():
        s = _score_symbol(sym, terms)
        if s > 0:
            scored[key] = s
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
    if ranker is not None and len(shortlist) > 1:
        try:
            snippets = [
                (getattr(symbols[k], "content", "") or getattr(symbols[k], "name", ""))[
                    :500
                ]
                for k, _ in shortlist
            ]
            order = ranker(query, snippets)
            if order:
                shortlist = [shortlist[i] for i in order if 0 <= i < len(shortlist)]
        except Exception:  # semantic rerank is best-effort; keep lexical order
            pass

    hits = []
    for key, score in shortlist[:top_k]:
        sym = symbols[key]
        hits.append(
            LocalizationHit(
                file_path=getattr(sym, "file_path", "") or "",
                symbol=getattr(sym, "name", "") or "",
                kind=getattr(sym, "kind", "") or "",
                score=round(score, 3),
            )
        )
    return hits


def localize_files(
    query: str, symbols: Dict[str, Any], *, top_k: int = 5, **kw
) -> List[Tuple[str, float]]:
    """File-level targets: sum symbol relevance per file, best-first. This is the
    shape decomposition wants — which files to scope a task to."""
    per_file: Dict[str, float] = {}
    for hit in localize(query, symbols, top_k=10_000, **kw):
        per_file[hit.file_path] = per_file.get(hit.file_path, 0.0) + hit.score
    return sorted(per_file.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
