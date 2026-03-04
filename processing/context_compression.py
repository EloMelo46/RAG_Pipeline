from __future__ import annotations

"""
Context compression — LLM extracts only the relevant parts from each chunk.
"""


def compress_context(llm, query: str, nodes: list) -> list[str]:
    """
    For each retrieved node, ask the LLM to extract only the
    information relevant to the query. This reduces noise and
    improves final answer quality.
    """
    compressed = []

    for i, node in enumerate(nodes):
        prompt = (
            "Extract ONLY the information relevant to the query below. "
            "Remove anything unrelated. Keep the original wording where possible.\n\n"
            f"Query:\n{query}\n\n"
            f"Text:\n{node.text}\n\n"
            "Relevant information:"
        )

        result = llm.complete(prompt)
        text = str(result).strip()

        if text:
            compressed.append(text)

    return compressed
