"""
vector_store.py
----------------
Stage 4 of the RAG pipeline: Vector Database.

A minimal, dependency-light vector store kept in memory as a NumPy array.
Supports adding embedded chunks and retrieving the top-k most similar
chunks to a query embedding via cosine similarity.

For larger corpora, swap this out for FAISS, Chroma, Pinecone, Weaviate,
etc. — the interface (add / search) is kept intentionally simple so that
swap is easy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class SearchResult:
    chunk_id: int
    text: str
    source: str
    score: float


class VectorStore:
    def __init__(self):
        self._embeddings: Optional[np.ndarray] = None  # shape (n, dim)
        self._chunk_ids: List[int] = []
        self._texts: List[str] = []
        self._sources: List[str] = []

    def add(self, embeddings: np.ndarray, chunks) -> None:
        """Add embedded chunks to the store.

        Args:
            embeddings: array of shape (n, dim), aligned 1:1 with `chunks`.
            chunks: list of Chunk objects (from chunking.py) with .chunk_id, .text, .source
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Number of chunks must match number of embedding rows")

        self._embeddings = embeddings if self._embeddings is None else np.vstack([self._embeddings, embeddings])
        for c in chunks:
            self._chunk_ids.append(c.chunk_id)
            self._texts.append(c.text)
            self._sources.append(c.source)

    def __len__(self) -> int:
        return len(self._texts)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[SearchResult]:
        """Return the top_k chunks most similar to query_embedding (cosine similarity)."""
        if self._embeddings is None or len(self._texts) == 0:
            return []

        q = query_embedding.reshape(1, -1)
        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        q_norm[q_norm == 0] = 1.0
        q_unit = q / q_norm

        m_norm = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        m_norm[m_norm == 0] = 1.0
        m_unit = self._embeddings / m_norm

        scores = (m_unit @ q_unit.T).flatten()  # cosine similarity, since both unit-normed

        top_k = min(top_k, len(scores))
        top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]  # sort the top_k by score desc

        return [
            SearchResult(
                chunk_id=self._chunk_ids[i],
                text=self._texts[i],
                source=self._sources[i],
                score=float(scores[i]),
            )
            for i in top_indices
        ]
