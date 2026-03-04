from __future__ import annotations

"""
Agentic retrieval — the LLM can decide to search multiple times.

Loop:
  1. Rewrite query → multi-query → vector search
  2. LLM reviews results → decides if more info is needed
  3. If yes → generates new query → repeats
  4. Final reranking over all accumulated candidates
"""

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from retrieval.query_rewriter import rewrite_query
from retrieval.multi_query import generate_multi_queries
from retrieval.retriever import retrieve_candidates
from retrieval.reranker import rerank

from config import FINAL_K, MAX_AGENT_STEPS


def agentic_retrieve(
    index: VectorStoreIndex,
    llm,
    query: str,
) -> list[NodeWithScore]:
    """
    Run an agentic retrieval loop:
    - Search, review, optionally search again.
    - Return the final top-K nodes after reranking.
    """
    all_candidates: list[NodeWithScore] = []
    current_query = query

    for step in range(MAX_AGENT_STEPS):
        step_label = f"  [Agent step {step + 1}/{MAX_AGENT_STEPS}]"

        # 1. Rewrite query
        rewritten = rewrite_query(llm, current_query)
        print(f"{step_label} Rewritten query: {rewritten[:80]}...")

        # 2. Generate multi-queries
        queries = generate_multi_queries(llm, rewritten)
        print(f"{step_label} Searching with {len(queries)} query variants...")

        # 3. Retrieve candidates
        candidates = retrieve_candidates(index, queries)
        all_candidates.extend(candidates)
        print(f"{step_label} Found {len(candidates)} candidates (total: {len(all_candidates)})")

        # 4. Ask LLM if more search is needed
        preview = "\n---\n".join([c.text[:200] for c in candidates[:5]])

        reasoning_prompt = (
            "You are helping retrieve documents for a question.\n\n"
            f"User question:\n{query}\n\n"
            f"Current retrieved snippets:\n{preview}\n\n"
            "Do we have enough relevant information to answer the question?\n\n"
            "If YES — respond with exactly: STOP\n"
            "If NO — respond with exactly ONE new search query to find "
            "the missing information."
        )

        decision = str(llm.complete(reasoning_prompt)).strip()

        if "STOP" in decision.upper():
            print(f"{step_label} Agent satisfied — stopping search.")
            break

        # Use the LLM's new query for the next loop
        current_query = decision
        print(f"{step_label} Agent wants more info: {current_query[:80]}...")

    # Deduplicate all accumulated candidates
    unique: dict[str, NodeWithScore] = {}
    for c in all_candidates:
        nid = c.node_id
        if nid not in unique or c.score > unique[nid].score:
            unique[nid] = c

    candidates = list(unique.values())
    print(f"  Total unique candidates: {len(candidates)}")

    # Final reranking
    top_nodes = rerank(query, candidates, FINAL_K)
    print(f"  After reranking: {len(top_nodes)} top results")

    return top_nodes
