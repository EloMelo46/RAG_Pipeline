"""
Ollama LLM client.
"""

from llama_index.llms.ollama import Ollama

from config import LLM_MODEL, LLM_TEMPERATURE


def get_llm() -> Ollama:
    """Create and return an Ollama LLM client."""
    return Ollama(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        request_timeout=120.0,
    )
