from __future__ import annotations

"""
Cross-encoder reranking using bge-reranker-large.
"""

from typing import Optional

from sentence_transformers import CrossEncoder
from llama_index.core.schema import NodeWithScore

from config import RERANK_MODEL

# Load once at module level (takes a few seconds on first import)
_reranker: Optional[CrossEncoder] = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print("  Loading reranker model...")
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def rerank(
    query: str,
    candidates: list[NodeWithScore],
    final_k: int,
) -> list[NodeWithScore]:
    """
    Re-score all candidates against the query using a cross-encoder,
    then return the top `final_k` results.
    """
    if not candidates:
        return []

    reranker = _get_reranker()

    texts = [c.text for c in candidates]
    pairs = [(query, t) for t in texts]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return [node for node, _score in ranked[:final_k]]
