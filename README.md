# RAG Pipeline

A modular, local RAG (Retrieval-Augmented Generation) pipeline built with **LlamaIndex**, **ChromaDB**, and **Ollama**.

## What It Does

You describe a goal → the system finds the most relevant documents from your knowledge base → reranks them → compresses the context → feeds everything to your local LLM → outputs the perfect answer.

**Everything runs locally.** No API keys, no cloud, no costs.

## Architecture

```
User Query
    ↓
Query Rewriting (LLM improves the search query)
    ↓
Multi-Query Generation (LLM creates 4 search variants)
    ↓
Vector Search (bge-large-en-v1.5 → ChromaDB)
    ↓
Agentic Retrieval Loop (LLM decides if more search is needed)
    ↓
Reranking (bge-reranker-large cross-encoder)
    ↓
Context Compression (LLM extracts only relevant parts)
    ↓
LLM Answer (Ollama)
```

## Quick Start

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.com/) with a model pulled (default: `gemma3:4b`)
- [Poetry](https://python-poetry.org/)

### Setup
```bash
# Install dependencies
poetry install

# Pull the LLM model (if not already done)
ollama pull gemma3:4b

# Run the pipeline (first run ingests data + downloads embedding models)
poetry run python main.py

# Subsequent runs (skip re-ingestion)
poetry run python main.py --skip-ingest
```

### Using a Different Data Source
```bash
# Override the data path
poetry run python main.py --data-path ./my_other_project
```

Or change `DATA_PATH` in `config.py` permanently.

## Configuration

All settings are in **`config.py`**:

| Setting | Default | Description |
|---------|---------|-------------|
| `DATA_PATH` | `./system-prompts-and-models-of-ai-tools-main` | Path to your documents |
| `EMBED_MODEL` | `BAAI/bge-large-en-v1.5` | Embedding model |
| `RERANK_MODEL` | `BAAI/bge-reranker-large` | Cross-encoder reranker |
| `LLM_MODEL` | `gemma3:4b` | Ollama model |
| `CHUNK_SIZE` | `800` | Tokens per chunk |
| `TOP_K` | `50` | Candidates from vector search |
| `FINAL_K` | `5` | Final results after reranking |
| `SYSTEM_PERSONA` | Prompt engineer | LLM persona / instructions |

### Changing the Persona

The `SYSTEM_PERSONA` in `config.py` controls how the LLM uses the retrieved context. Change it to suit your use case:

- **Prompt Engineer** (default): Generates optimal system prompts
- **Code Assistant**: Answers coding questions from your codebase
- **Documentation Q&A**: Answers questions from your docs

## Project Structure

```
RAG_Pipeline/
├── main.py                  # Entry point
├── config.py                # All settings
├── pyproject.toml            # Poetry dependencies
├── loaders/
│   └── loader.py            # Universal file loaders
├── indexing/
│   ├── chunking.py          # SentenceSplitter
│   └── build_index.py       # ChromaDB + embeddings
├── retrieval/
│   ├── query_rewriter.py    # LLM query rewriting
│   ├── multi_query.py       # Multi-query generation
│   ├── retriever.py         # Vector retrieval
│   ├── reranker.py          # Cross-encoder reranking
│   └── agentic_retriever.py # Agentic retrieval loop
├── processing/
│   └── context_compression.py  # Context compression
└── llm/
    └── ollama_client.py     # Ollama client
```

## Supported File Types

`.txt` `.md` `.json` `.csv` `.pdf` `.html` `.py` `.js` `.ts` `.yaml` `.toml`
