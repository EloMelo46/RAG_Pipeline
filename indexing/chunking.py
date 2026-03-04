from __future__ import annotations

"""
Document chunking using LlamaIndex SentenceSplitter.
"""

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode

from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(documents: list[Document]) -> list[BaseNode]:
    """
    Split documents into chunks using SentenceSplitter.

    - Respects sentence boundaries when possible.
    - chunk_size and chunk_overlap are configured in config.py.
    """
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    nodes = splitter.get_nodes_from_documents(documents)
    return nodes
