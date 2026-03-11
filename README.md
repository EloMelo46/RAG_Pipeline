# RAG Pipeline

A modular, local RAG (Retrieval-Augmented Generation) pipeline built with **LlamaIndex**, **ChromaDB**, and **Ollama**.

## What It Does

You describe a goal → the system finds the most relevant documents from your knowledge base → reranks them → feeds everything to your local LLM → outputs the answer.

**Everything runs locally.** No API keys, no cloud, no costs.

## Architecture

```
User Query
    ↓
Multi-Query Generation (LLM creates N search variants)
    ↓
Vector Search (embedding model → ChromaDB)
    ↓
Agentic Retrieval Loop (LLM decides if more search is needed)
    ↓
Reranking (cross-encoder reranker)
    ↓
[Optional] Context Compression (LLM extracts only relevant parts)
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
# Override the data path at runtime
poetry run python main.py --data-path ./my_other_project
```

Or change `DATA_PATH` in `config.py` permanently.

## Configuration

All settings are in **`config.py`**:

| Setting | Default | Description |
|---------|---------|-------------|
| `DATA_PATH` | `./system-prompts-...` | Path to your documents |
| `EMBED_MODEL` | `BAAI/bge-base-en-v1.5` | Embedding model |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder reranker |
| `LLM_MODEL` | `gemma3:4b` | Ollama model |
| `CHUNK_SIZE` | `800` | Tokens per chunk |
| `TOP_K` | `10` | Candidates from vector search |
| `FINAL_K` | `3` | Final results after reranking |
| `MAX_AGENT_STEPS` | `1` | Max agentic retrieval loops |
| `MULTI_QUERY_VARIANTS` | `3` | Search variants (1 = single query, no LLM call) |
| `USE_CONTEXT_COMPRESSION` | `False` | Extract relevant parts per chunk (slower) |
| `ENABLE_INPUT_GUARDRAILS` | `True` | Run Llama Guard on every user query before retrieval |
| `ENABLE_OUTPUT_GUARDRAILS` | `True` | Run Llama Guard on every LLM response before returning it |
| `GUARDRAIL_MODEL` | `llama-guard3:1b` | Ollama model used for safety checks |
| `SYSTEM_PERSONA` | Prompt engineer | LLM persona / instructions |

## Performance Tuning

Every query passes through several steps, each with a speed/quality tradeoff:

### Speed presets

**Maximum speed** (minimal LLM calls):
```python
MULTI_QUERY_VARIANTS = 1      # No multi-query LLM call
MAX_AGENT_STEPS = 1           # Single search pass
USE_CONTEXT_COMPRESSION = False
TOP_K = 10
FINAL_K = 3
EMBED_MODEL = "BAAI/bge-small-en-v1.5"    # 130 MB
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # 80 MB
```

**Balanced** (default):
```python
MULTI_QUERY_VARIANTS = 3
MAX_AGENT_STEPS = 1
USE_CONTEXT_COMPRESSION = False
TOP_K = 10
FINAL_K = 3
EMBED_MODEL = "BAAI/bge-base-en-v1.5"    # 440 MB
RERANK_MODEL = "BAAI/bge-reranker-base"  # 280 MB
```

**Maximum quality** (recommended with 7B+ LLM):
```python
MULTI_QUERY_VARIANTS = 5
MAX_AGENT_STEPS = 2
USE_CONTEXT_COMPRESSION = True
TOP_K = 50
FINAL_K = 5
EMBED_MODEL = "BAAI/bge-large-en-v1.5"      # 1.3 GB
RERANK_MODEL = "BAAI/bge-reranker-large"     # 2.2 GB
LLM_MODEL = "qwen2.5:7b"
```

### What makes it slow

| Step | LLM Calls | Impact |
|------|-----------|--------|
| Multi-Query (`MULTI_QUERY_VARIANTS > 1`) | 1× | Medium |
| Agent decision (`MAX_AGENT_STEPS`) | 1× per step | Medium |
| Context Compression (`USE_CONTEXT_COMPRESSION = True`) | 1× per chunk | **High** |
| Large embedding model | — | First run only |
| Large reranker model | — | Every query |

> **Note:** If you change `EMBED_MODEL`, delete `chroma_db/` and re-ingest. Vectors are model-specific.

## Security Guardrails

The pipeline uses [Llama Guard 3](https://ollama.com/library/llama-guard3) as a dedicated safety model to screen inputs and outputs.

```bash
# Pull the guard model (one-time, ~1.6 GB)
ollama pull llama-guard3:1b
```

Two checks run on every query:

1. **Input check** — before retrieval starts. Catches prompt injections, jailbreak attempts, and policy violations. Blocked queries never touch the vector database or the main LLM.
2. **Output check** — after the main LLM generates its answer. Catches cases where the model produces an unsafe response despite a benign-looking query.

Both checks return either `safe` or `unsafe <category>` (e.g. `S13` for prompt injection). If unsafe, a plain error message is returned instead of the original response.

Disable either check in `config.py` when speed matters more than safety:

```python
ENABLE_INPUT_GUARDRAILS = False
ENABLE_OUTPUT_GUARDRAILS = False
```

Swap the model at any time without touching the pipeline code:

```python
GUARDRAIL_MODEL = "llama-guard3:8b"  # higher accuracy, more VRAM
```

### Changing the Persona

Set `SYSTEM_PERSONA` in `config.py` to adapt the pipeline to any domain:

- **Prompt Engineer** (default): Generates optimal system prompts
- **Code Assistant**: Answers coding questions from your codebase
- **Documentation Q&A**: Answers questions from your docs
- **Any custom role**: Just describe it in plain text

## Project Structure

```
RAG_Pipeline/
├── main.py                      # Entry point
├── config.py                    # All settings
├── test_security.py             # Security guardrails smoke test
├── pyproject.toml               # Poetry dependencies
├── loaders/loader.py            # Universal file loaders
├── indexing/
│   ├── chunking.py              # SentenceSplitter
│   └── build_index.py           # ChromaDB + embeddings
├── retrieval/
│   ├── multi_query.py           # Multi-query generation
│   ├── retriever.py             # Vector retrieval
│   ├── reranker.py              # Cross-encoder reranking
│   └── agentic_retriever.py     # Agentic retrieval loop
├── processing/
│   └── context_compression.py   # Context compression
├── security/
│   ├── input_guardrails.py      # Llama Guard input check
│   └── output_guardrails.py     # Llama Guard output check
└── llm/ollama_client.py         # Ollama client
```

## Supported File Types

`.txt` `.md` `.json` `.csv` `.pdf` `.html` `.py` `.js` `.ts` `.yaml` `.toml`
