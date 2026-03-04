"""
Query rewriting — LLM improves the user's query for better retrieval.
"""


def rewrite_query(llm, query: str) -> str:
    """
    Ask the LLM to rephrase the query so it works better
    for semantic document retrieval.
    """
    prompt = (
        "Rewrite the following query so it is optimized for semantic "
        "document retrieval. Make it more specific and descriptive.\n\n"
        f"Query:\n{query}\n\n"
        "Return ONLY the improved query, nothing else."
    )

    result = llm.complete(prompt)
    rewritten = str(result).strip()

    # Fallback: if the LLM returns nothing useful, use the original
    if not rewritten or len(rewritten) < 5:
        return query

    return rewritten
