from __future__ import annotations

"""
Build and load a VectorStoreIndex backed by ChromaDB.
"""

import chromadb

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import EMBED_MODEL, CHROMA_PATH, COLLECTION_NAME


def _get_embed_model() -> HuggingFaceEmbedding:
    """Create the HuggingFace embedding model (downloaded on first use)."""
    return HuggingFaceEmbedding(model_name=EMBED_MODEL)


def _get_chroma_store() -> ChromaVectorStore:
    """Get or create a persistent ChromaDB collection."""
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = db.get_or_create_collection(COLLECTION_NAME)
    return ChromaVectorStore(chroma_collection=collection)


def build_index(nodes: list[BaseNode]) -> VectorStoreIndex:
    """
    Build a new VectorStoreIndex from document nodes.

    - Embeds every node using bge-large-en-v1.5.
    - Stores vectors persistently in ChromaDB.
    """
    embed_model = _get_embed_model()
    vector_store = _get_chroma_store()

    index = VectorStoreIndex(
        nodes,
        embed_model=embed_model,
        vector_store=vector_store,
    )

    return index


def load_existing_index() -> VectorStoreIndex:
    """
    Load an existing VectorStoreIndex from a previously built ChromaDB.

    Use this to skip re-ingestion when the data hasn't changed.
    """
    embed_model = _get_embed_model()
    vector_store = _get_chroma_store()

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    return index
