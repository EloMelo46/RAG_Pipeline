from __future__ import annotations

"""
Universal file loaders.

Supports: .txt, .md, .py, .js, .json, .csv, .pdf, .html/.htm
Each loader returns a list of LlamaIndex Document objects.
"""

import os
import json

import pandas as pd
import pdfplumber
from bs4 import BeautifulSoup
from llama_index.core import Document

from config import DATA_PATH


# ---------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------

def load_txt(path: str) -> list[Document]:
    """Load plain text files (.txt, .md, .py, .js, etc.)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if not text.strip():
        return []
    return [Document(text=text, metadata={"source": os.path.relpath(path, DATA_PATH)})]


def load_json(path: str) -> list[Document]:
    """Load JSON files — each top-level list item becomes its own Document."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        return load_txt(path)  # Fallback: treat as plain text

    rel = os.path.relpath(path, DATA_PATH)
    docs = []

    if isinstance(data, list):
        for i, obj in enumerate(data):
            docs.append(Document(
                text=json.dumps(obj, indent=2, ensure_ascii=False),
                metadata={"source": rel, "item_index": i},
            ))
    else:
        docs.append(Document(
            text=json.dumps(data, indent=2, ensure_ascii=False),
            metadata={"source": rel},
        ))

    return docs


def load_csv(path: str) -> list[Document]:
    """Load CSV files — each row becomes a JSON Document."""
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        return []

    rel = os.path.relpath(path, DATA_PATH)
    docs = []

    for i, row in df.iterrows():
        docs.append(Document(
            text=row.to_json(force_ascii=False),
            metadata={"source": rel, "row_index": i},
        ))

    return docs


def load_pdf(path: str) -> list[Document]:
    """Load PDF files — each page becomes a Document."""
    docs = []
    rel = os.path.relpath(path, DATA_PATH)

    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    docs.append(Document(
                        text=text,
                        metadata={"source": rel, "page": i + 1},
                    ))
    except Exception:
        pass

    return docs


def load_html(path: str) -> list[Document]:
    """Load HTML files — extract text content."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
    except Exception:
        return []

    if not text.strip():
        return []

    return [Document(
        text=text,
        metadata={"source": os.path.relpath(path, DATA_PATH)},
    )]


# ---------------------------------------------------------
# File router
# ---------------------------------------------------------

EXTENSION_MAP = {
    ".txt": load_txt,
    ".md": load_txt,
    ".py": load_txt,
    ".js": load_txt,
    ".ts": load_txt,
    ".yaml": load_txt,
    ".yml": load_txt,
    ".toml": load_txt,
    ".cfg": load_txt,
    ".ini": load_txt,
    ".json": load_json,
    ".csv": load_csv,
    ".pdf": load_pdf,
    ".html": load_html,
    ".htm": load_html,
}


def load_file(path: str) -> list[Document]:
    """Route a file to the correct loader based on its extension."""
    ext = os.path.splitext(path)[1].lower()
    loader = EXTENSION_MAP.get(ext)

    if loader is None:
        return []

    return loader(path)


# ---------------------------------------------------------
# Load entire project
# ---------------------------------------------------------

def load_project(data_path: str = None) -> list[Document]:
    """
    Recursively walk `data_path` and load all supported files.
    Returns a flat list of LlamaIndex Document objects.
    """
    root = data_path or DATA_PATH
    documents = []

    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            docs = load_file(full_path)
            documents.extend(docs)

    return documents
