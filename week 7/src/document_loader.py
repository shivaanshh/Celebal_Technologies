"""
document_loader.py
-------------------
Stage 1 of the RAG pipeline: Document Ingestion.

Responsible for loading raw documents (PDF, TXT, MD) from disk and
converting them into plain text that downstream stages (chunking,
embedding) can consume.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List


@dataclass
class LoadedDocument:
    """A single ingested document."""
    source_path: str
    text: str
    num_chars: int


class DocumentLoadError(Exception):
    """Raised when a document cannot be read or parsed."""


def _load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _load_pdf(path: str) -> str:
    # Try pymupdf first — handles far more PDF types (complex encoding, multi-column, etc.)
    try:
        import fitz  # pymupdf
        doc = fitz.open(path)
        pages_text = [page.get_text() for page in doc]
        doc.close()
        text = "\n".join(pages_text)
        if text.strip():
            return text
        # Falls through to pypdf if fitz extracted nothing (e.g. pure-image PDF)
    except ImportError:
        pass
    except Exception as exc:
        print(f"[info] pymupdf failed for '{path}' ({exc}), trying pypdf…")

    # Fallback to pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        pages_text = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages_text)
    except ImportError as exc:
        raise DocumentLoadError(
            "No PDF library available. Install pymupdf: pip install pymupdf"
        ) from exc
    except Exception as exc:
        raise DocumentLoadError(f"Failed to read PDF '{path}': {exc}") from exc


_LOADERS = {
    ".txt": _load_txt,
    ".md": _load_txt,
    ".pdf": _load_pdf,
}


def load_document(path: str) -> LoadedDocument:
    """Load a single document from disk into a LoadedDocument."""
    if not os.path.isfile(path):
        raise DocumentLoadError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise DocumentLoadError(
            f"Unsupported file type '{ext}'. Supported types: {list(_LOADERS.keys())}"
        )

    raw_text = loader(path)
    cleaned = _clean_text(raw_text)

    if not cleaned.strip():
        raise DocumentLoadError(f"No extractable text found in: {path}")

    return LoadedDocument(source_path=path, text=cleaned, num_chars=len(cleaned))


def load_documents(paths: List[str]) -> List[LoadedDocument]:
    """Load multiple documents, skipping ones that fail with a warning."""
    docs = []
    for p in paths:
        try:
            docs.append(load_document(p))
        except DocumentLoadError as e:
            print(f"[warning] Skipping '{p}': {e}")
    return docs


def _clean_text(text: str) -> str:
    """Light normalization: collapse excessive whitespace/newlines."""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)
