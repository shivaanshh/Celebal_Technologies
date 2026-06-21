"""
chunking.py
-----------
Stage 2 of the RAG pipeline: Text Chunking.

Splits long document text into smaller, overlapping chunks. Chunking
improves retrieval accuracy because embedding models work best on
short, semantically coherent passages rather than entire documents.

Strategy used here: sentence-aware chunking. We split on sentence
boundaries first, then greedily pack sentences into chunks up to
`chunk_size` words, with `overlap` words repeated between consecutive
chunks so context isn't lost at chunk boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    chunk_id: int
    text: str
    source: str
    word_count: int


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_into_sentences(text: str) -> List[str]:
    # Split paragraph-by-paragraph first to keep structure, then sentences.
    sentences = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        for sent in _SENTENCE_SPLIT_RE.split(para):
            sent = sent.strip()
            if sent:
                sentences.append(sent)
    return sentences


def chunk_text(
    text: str,
    source: str = "document",
    chunk_size: int = 150,
    overlap: int = 30,
) -> List[Chunk]:
    """
    Split `text` into overlapping chunks of roughly `chunk_size` words.

    Args:
        text: Raw document text.
        source: Identifier (e.g. filename) stored with each chunk for citation.
        chunk_size: Target number of words per chunk.
        overlap: Number of words repeated between consecutive chunks.

    Returns:
        List of Chunk objects.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    sentences = split_into_sentences(text)

    chunks: List[Chunk] = []
    current_words: List[str] = []
    chunk_id = 0

    def flush_chunk():
        nonlocal chunk_id
        if current_words:
            chunk_text_str = " ".join(current_words)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text_str,
                    source=source,
                    word_count=len(current_words),
                )
            )
            chunk_id += 1

    for sentence in sentences:
        words = sentence.split()
        if len(current_words) + len(words) > chunk_size and current_words:
            flush_chunk()
            # keep last `overlap` words for context continuity
            current_words = current_words[-overlap:] if overlap > 0 else []
        current_words.extend(words)

    flush_chunk()  # flush any remainder

    return chunks


def chunk_documents(loaded_documents, chunk_size: int = 150, overlap: int = 30) -> List[Chunk]:
    """Chunk a list of LoadedDocument objects into a single flat list of Chunks
    with globally unique chunk_ids."""
    all_chunks: List[Chunk] = []
    next_id = 0
    for doc in loaded_documents:
        doc_chunks = chunk_text(
            doc.text, source=doc.source_path, chunk_size=chunk_size, overlap=overlap
        )
        for c in doc_chunks:
            c.chunk_id = next_id
            next_id += 1
            all_chunks.append(c)
    return all_chunks
