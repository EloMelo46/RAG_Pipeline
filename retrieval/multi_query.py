from __future__ import annotations

"""
Multi-query generation — LLM creates multiple search query variants.
"""


def generate_multi_queries(llm, query: str, num_variants: int = 4) -> list[str]:
    """
    Generate alternative search queries to increase retrieval recall.

    Returns the original query plus `num_variants` alternatives.
    """
    prompt = (
        f"Generate {num_variants} alternative search queries for retrieving "
        "relevant documents. Each query should approach the topic from a "
        "different angle.\n\n"
        f"Original query:\n{query}\n\n"
        "Return ONLY the queries, one per line, no numbering or bullets."
    )

    response = llm.complete(prompt)
    lines = str(response).strip().split("\n")

    # Clean up and filter empty lines
    variants = [line.strip().strip("- ").strip() for line in lines]
    variants = [v for v in variants if v and len(v) > 5]

    # Always include the original query first
    return [query] + variants
