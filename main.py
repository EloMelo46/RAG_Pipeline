"""
RAG Pipeline — Main Entry Point

Usage:
    poetry run python main.py              # First run: ingests data + starts interactive loop
    poetry run python main.py --skip-ingest # Skip ingestion, use existing index
"""

import argparse
import os
import sys

from config import DATA_PATH, CHROMA_PATH, SYSTEM_PERSONA, USE_CONTEXT_COMPRESSION
from loaders.loader import load_project
from indexing.chunking import chunk_documents
from indexing.build_index import build_index, load_existing_index
from retrieval.agentic_retriever import agentic_retrieve
from processing.context_compression import compress_context
from llm.ollama_client import get_llm


def ingest(data_path: str = None):
    """Load documents, chunk them, and build the vector index."""
    path = data_path or DATA_PATH

    print(f"\nLoading documents from: {path}")
    docs = load_project(path)
    print(f"   Documents loaded: {len(docs)}")

    if not docs:
        print("   No documents found! Check DATA_PATH in config.py.")
        sys.exit(1)

    print(f"\nChunking documents...")
    nodes = chunk_documents(docs)
    print(f"   Chunks created: {len(nodes)}")

    print(f"\nBuilding vector index (this may take a while on first run)...")
    index = build_index(nodes)
    print(f"   Index built and stored in: {CHROMA_PATH}")

    return index


def ask(index, llm, query: str) -> str:
    """
    Full RAG pipeline:
      1. Agentic retrieval (rewrite → multi-query → search → decide → repeat)
      2. Context compression
      3. LLM generates final answer
    """
    print("\nRunning agentic retrieval...")
    top_nodes = agentic_retrieve(index, llm, query)

    if not top_nodes:
        return "No relevant documents found. Try rephrasing your query."

    print("\nCompressing context..." if USE_CONTEXT_COMPRESSION else "\nUsing raw chunks...")
    if USE_CONTEXT_COMPRESSION:
        compressed = compress_context(llm, query, top_nodes)
    else:
        compressed = [node.text for node in top_nodes]

    context = "\n\n---\n\n".join(compressed)

    prompt = (
        f"{SYSTEM_PERSONA}\n\n"
        f"<context>\n{context}\n</context>\n\n"
        f"User goal:\n{query}\n\n"
        "Your response:"
    )

    print("\nGenerating answer...")
    response = llm.complete(prompt)

    return str(response)


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline")
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip data ingestion and use the existing ChromaDB index.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override DATA_PATH from config.py.",
    )
    args = parser.parse_args()

    # ── Build or load index ──────────────────────────────────
    if args.skip_ingest:
        if not os.path.exists(CHROMA_PATH):
            print("No existing index found. Run without --skip-ingest first.")
            sys.exit(1)
        print("Loading existing index from ChromaDB...")
        index = load_existing_index()
        print("   Index loaded.")
    else:
        index = ingest(args.data_path)

    # ── Connect to Ollama ────────────────────────────────────
    print("\nConnecting to Ollama LLM...")
    llm = get_llm()
    print("   LLM ready.\n")

    # ── Interactive loop ─────────────────────────────────────
    print("=" * 60)
    print("  RAG Pipeline — Interactive Mode")
    print("  Type your query, or 'exit' to quit.")
    print("=" * 60)

    while True:
        try:
            query = input("\nQuery: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q"):
            print("Bye!")
            break

        answer = ask(index, llm, query)

        print("\n" + "─" * 60)
        print(answer)
        print("─" * 60)


if __name__ == "__main__":
    main()
