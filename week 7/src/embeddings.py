"""
embeddings.py
-------------
Stage 3 of the RAG pipeline: Embedding Creation.

Converts text chunks into vector representations that capture semantic
meaning, so that similarity search (Stage 4/6) can find relevant chunks
for a given query.

Two backends are supported:

1. SentenceTransformerBackend (preferred): uses a pretrained sentence
   embedding model (e.g. 'all-MiniLM-L6-v2') from the `sentence-transformers`
   library. Produces dense semantic embeddings. Requires internet access
   to download model weights the first time, and the `sentence-transformers`
   package to be installed.

2. TfidfBackend (automatic fallback): uses scikit-learn's TF-IDF
   vectorizer + SVD-based dimensionality reduction. Works fully offline
   with no model downloads. Captures lexical/keyword similarity rather
   than deep semantic similarity, but is a solid, dependency-light
   baseline and a faithful illustration of "text -> vector" retrieval.

The EmbeddingModel class auto-selects the best available backend, so the
rest of the pipeline doesn't need to know which one is active.
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np


class EmbeddingModel:
    """Unified interface for turning text into vectors.

    Usage:
        model = EmbeddingModel()
        model.fit(corpus_texts)          # required for TF-IDF backend, no-op for ST
        vectors = model.encode(texts)    # -> np.ndarray shape (n, dim)
    """

    def __init__(self, prefer_sentence_transformers: bool = True, model_name: str = "all-MiniLM-L6-v2"):
        self.backend_name: str
        self._st_model = None
        self._tfidf_vectorizer = None
        self._svd = None

        if prefer_sentence_transformers:
            try:
                from sentence_transformers import SentenceTransformer

                self._st_model = SentenceTransformer(model_name)
                self.backend_name = f"sentence-transformers ({model_name})"
                return
            except Exception as exc:
                print(
                    f"[info] sentence-transformers backend unavailable ({exc.__class__.__name__}: {exc}). "
                    "Falling back to local TF-IDF embeddings."
                )

        self.backend_name = "tfidf"

    # ------------------------------------------------------------------
    def fit(self, corpus_texts: List[str]) -> None:
        """Fit any backend that requires a corpus-level vocabulary (TF-IDF).
        No-op for sentence-transformers, which uses a pretrained model."""
        if self._st_model is not None:
            return

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self._tfidf_vectorizer = TfidfVectorizer(
            stop_words="english", max_features=20000, ngram_range=(1, 2)
        )
        tfidf_matrix = self._tfidf_vectorizer.fit_transform(corpus_texts)

        # Reduce dimensionality so vectors behave like dense embeddings
        # (also speeds up similarity search). Cap components by corpus size
        # so we never request more components than the data can support
        # (avoids numerically unstable SVD fits on tiny corpora).
        n_features, n_samples = tfidf_matrix.shape[1], tfidf_matrix.shape[0]
        n_components = max(1, min(256, n_features - 1, n_samples - 1))
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        with warnings.catch_warnings():
            # On very small/low-rank corpora, sklearn emits a harmless
            # RuntimeWarning computing its internal explained_variance_ratio_
            # metric, which we never use. Suppress just that noise.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self._svd.fit(tfidf_matrix)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into an (n, dim) array of embeddings."""
        if self._st_model is not None:
            return np.asarray(
                self._st_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            )

        if self._tfidf_vectorizer is None or self._svd is None:
            raise RuntimeError("TF-IDF backend not fitted yet. Call .fit(corpus_texts) first.")

        tfidf_matrix = self._tfidf_vectorizer.transform(texts)
        dense = self._svd.transform(tfidf_matrix)
        # L2-normalize so cosine similarity == dot product
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense / norms
