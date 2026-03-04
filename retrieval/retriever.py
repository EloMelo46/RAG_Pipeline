from __future__ import annotations

"""
Vector retrieval — search the index for each query and deduplicate.
"""

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from config import TOP_K


def retrieve_candidates(
    index: VectorStoreIndex,
    queries: list[str],
) -> list[NodeWithScore]:
    """
    Run vector search for each query, then deduplicate by node_id.

    Returns a flat list of unique candidate nodes.
    """
    retriever = index.as_retriever(similarity_top_k=TOP_K)

    all_results: list[NodeWithScore] = []

    for q in queries:
        try:
            results = retriever.retrieve(q)
            all_results.extend(results)
        except Exception:
            continue

    # Deduplicate by node_id, keeping highest-scoring occurrence
    unique: dict[str, NodeWithScore] = {}
    for r in all_results:
        nid = r.node_id
        if nid not in unique or r.score > unique[nid].score:
            unique[nid] = r

    return list(unique.values())
