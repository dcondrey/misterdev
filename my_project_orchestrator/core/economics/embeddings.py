"""Semantic relevance ranking for context selection.

When a task pulls in more candidate code symbols than fit the context budget,
the symbols were previously truncated by arbitrary order. This ranks candidates
by cosine similarity of their embedding to the task description and keeps the
most relevant — the lossless-relevant version of "fit more useful information
into context".

Everything here degrades gracefully: any embedding failure falls back to the
prior arbitrary-order selection, so semantic ranking can change *which* context
is shown but never break a build. Embeddings of unchanged code are stable, so
vectors are cached on disk (keyed by model + text hash) and reused across runs.
"""

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Set

from my_project_orchestrator.logging_setup import setup_logger
from my_project_orchestrator.utils.file_utils import ensure_artifact_dir

logger = setup_logger(__name__)

# Splits identifiers into subword tokens: snake_case (the separators aren't
# matched) and camelCase boundaries, plus numbers. Single-char tokens dropped.
_TOKEN = re.compile(r"[A-Za-z][a-z0-9]*|[A-Z]+(?![a-z])|[0-9]+")


def _tokenize(text: str) -> Set[str]:
    return {t.lower() for t in _TOKEN.findall(text or "") if len(t) > 1}


def _lexical_overlap(query_tokens: Set[str], text_tokens: Set[str]) -> float:
    """Fraction of query identifier tokens present in the candidate (0..1).

    Recall-of-query rather than Jaccard, so a large candidate that contains the
    query's identifiers isn't penalized for its size — the right bias for code,
    where exact identifier matches are strong relevance signals.
    """
    if not query_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


_EMBEDDING_MODELS_URL = "https://openrouter.ai/api/v1/embeddings/models"


def _fetch_embedding_models() -> list:
    from my_project_orchestrator.core.economics.free_models import _http_fetch

    return _http_fetch(_EMBEDDING_MODELS_URL)


def pick_embedding_model(
    configured: str, prefer: Optional[List[str]] = None, fetcher=None
) -> Optional[str]:
    """The embedding model to use: the explicit config value, else discovered.

    Mirrors free-model harvesting — discovers OpenRouter's embedding models and
    picks the cheapest (free sorts first). Among equally-priced models, prefers
    ones whose id matches a ``prefer`` hint (default ['code'], since code-aware
    embeddings rank code better); ties broken deterministically by id. Returns
    None if nothing is configured and discovery fails, so semantic retrieval
    stays off gracefully rather than guessing a model.
    """
    if configured:
        return configured
    hints = [h.lower() for h in (prefer if prefer is not None else ["code"])]
    try:
        models = (fetcher or _fetch_embedding_models)()
    except Exception as e:
        logger.warning(f"Embedding-model discovery failed: {e}")
        return None
    ranked = []
    for m in models:
        if not isinstance(m, dict) or not m.get("id"):
            continue
        pricing = m.get("pricing") or {}
        try:
            price = float(pricing.get("prompt", "inf"))
        except (TypeError, ValueError):
            price = float("inf")
        model_id = m["id"]
        preferred = 0 if any(h in model_id.lower() for h in hints) else 1
        ranked.append((price, preferred, model_id))
    if not ranked:
        return None
    ranked.sort(key=lambda r: (r[0], r[1], r[2]))
    logger.info(f"Auto-selected embedding model {ranked[0][2]!r}")
    return ranked[0][2]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 on degenerate input)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = math.fsum(x * y for x, y in zip(a, b))
    na = math.sqrt(math.fsum(x * x for x in a))
    nb = math.sqrt(math.fsum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class Embedder(Protocol):
    """Minimal interface a semantic ranker needs from an embedding client."""

    model: str

    def embed(self, texts: List[str]) -> List[List[float]]: ...


class EmbeddingCache:
    """Disk-backed cache of text -> vector, keyed by model and content hash."""

    def __init__(self, path: Path, model: str):
        self.path = Path(path)
        self.model = model
        self._vectors: Optional[Dict[str, List[float]]] = None

    def _key(self, text: str) -> str:
        h = hashlib.sha256()
        h.update(f"{self.model}\x00".encode("utf-8"))
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def _load(self) -> Dict[str, List[float]]:
        if self._vectors is None:
            self._vectors = {}
            if self.path.exists():
                try:
                    data = json.loads(self.path.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        self._vectors = data
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Embedding cache unreadable, starting fresh: {e}")
        return self._vectors

    def get(self, text: str) -> Optional[List[float]]:
        return self._load().get(self._key(text))

    def put_many(self, items: Dict[str, List[float]]) -> None:
        """Store {text: vector} and persist."""
        store = self._load()
        for text, vector in items.items():
            store[self._key(text)] = vector
        ensure_artifact_dir(self.path.parent)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(store), encoding="utf-8")
        tmp.replace(self.path)


class SemanticRanker:
    """Selects the top-k most task-relevant items via a hybrid relevance score.

    Combines dense embedding cosine with a lexical identifier-overlap signal —
    code relevance is driven heavily by exact identifier matches, which dense
    similarity alone can miss. The embedder is optional: with none (or on an
    embedding failure) the ranker degrades to lexical-only, which still beats the
    arbitrary-order slice it replaces.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        cache: Optional[EmbeddingCache] = None,
        lexical_weight: float = 0.3,
    ):
        self.embedder = embedder
        self.cache = cache
        self.lexical_weight = max(0.0, min(1.0, lexical_weight))

    def _embed_with_cache(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Embed texts, serving cache hits and embedding only the misses."""
        cached: Dict[str, Optional[List[float]]] = {}
        misses: List[str] = []
        for text in texts:
            vec = self.cache.get(text) if self.cache else None
            cached[text] = vec
            if vec is None and text not in misses:
                misses.append(text)
        if misses:
            fresh = self.embedder.embed(misses)
            if len(fresh) != len(misses):
                raise ValueError("embedding count mismatch")
            new = dict(zip(misses, fresh))
            if self.cache:
                self.cache.put_many(new)
            for text, vec in new.items():
                cached[text] = vec
        return [cached[text] for text in texts]

    def top_k(self, query: str, candidates: Dict[str, str], k: int) -> List[str]:
        """Return up to k candidate ids most relevant to query.

        candidates maps id -> text. Scores blend dense cosine (when an embedder
        is available) with lexical identifier overlap; on embedding failure it
        uses lexical alone. Never raises — ranking is an optimization under the
        validation gates, not load-bearing for correctness.
        """
        ids = list(candidates)
        if k >= len(ids):
            return ids

        query_tokens = _tokenize(query)
        lexical = {
            i: _lexical_overlap(query_tokens, _tokenize(candidates[i])) for i in ids
        }

        dense: Optional[Dict[str, float]] = None
        if self.embedder is not None:
            try:
                vectors = self._embed_with_cache([query] + [candidates[i] for i in ids])
                query_vec = vectors[0]
                dense = {
                    i: (cosine_similarity(query_vec, v) + 1.0) / 2.0
                    for i, v in zip(ids, vectors[1:])
                }
            except Exception as e:
                logger.warning(f"Dense ranking unavailable, using lexical only: {e}")

        if dense is None:

            def score(i: str) -> float:
                return lexical[i]
        else:
            w = self.lexical_weight

            def score(i: str) -> float:
                return (1.0 - w) * dense[i] + w * lexical[i]

        return sorted(ids, key=score, reverse=True)[:k]
