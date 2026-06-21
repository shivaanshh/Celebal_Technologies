"""
pipeline.py
-----------
Ties together all 7 stages of the RAG pipeline into a single, simple
interface:

    1. Document Ingestion   -> document_loader.py
    2. Text Chunking        -> chunking.py
    3. Embedding Creation   -> embeddings.py
    4. Vector Database      -> vector_store.py
    5. Query Processing     -> embeddings.py (re-used for the query)
    6. Context Retrieval    -> vector_store.py
    7. Answer Generation    -> generator.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .document_loader import load_documents
from .chunking import chunk_documents, Chunk
from .embeddings import EmbeddingModel
from .vector_store import VectorStore, SearchResult
from .generator import AnswerGenerator


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: List[SearchResult]


class RAGPipeline:
    def __init__(
        self,
        chunk_size: int = 150,
        chunk_overlap: int = 30,
        top_k: int = 3,
        llm_backend: str = "auto",
        llm_model: str = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()
        self.generator = AnswerGenerator(backend=llm_backend, model=llm_model)

        self._chunks: List[Chunk] = []
        self._is_built = False

    # ------------------------------------------------------------------
    def ingest(self, paths: List[str]) -> None:
        """Stages 1-4: load documents, chunk them, embed them, store them."""
        print(f"[1/4] Loading {len(paths)} document(s)...")
        docs = load_documents(paths)
        if not docs:
            raise RuntimeError("No documents were successfully loaded.")
        total_chars = sum(d.num_chars for d in docs)
        print(f"      Loaded {len(docs)} document(s), {total_chars:,} characters total.")

        print(f"[2/4] Chunking text (chunk_size={self.chunk_size} words, overlap={self.chunk_overlap})...")
        self._chunks = chunk_documents(docs, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
        print(f"      Produced {len(self._chunks)} chunks.")

        print(f"[3/4] Creating embeddings (backend: {self.embedding_model.backend_name})...")
        chunk_texts = [c.text for c in self._chunks]
        self.embedding_model.fit(chunk_texts)
        chunk_embeddings = self.embedding_model.encode(chunk_texts)

        print(f"[4/4] Storing {len(self._chunks)} embeddings in the vector store...")
        self.vector_store.add(chunk_embeddings, self._chunks)
        self._is_built = True
        print("Ingestion complete. Ready for questions.\n")

    def add_text(self, text: str, source: str = "inline_text") -> None:
        """Convenience method to ingest raw text directly (no file on disk)."""
        from .document_loader import LoadedDocument

        doc = LoadedDocument(source_path=source, text=text, num_chars=len(text))
        self._ingest_loaded_docs([doc])

    def _ingest_loaded_docs(self, docs) -> None:
        new_chunks = chunk_documents(docs, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
        # offset chunk_ids to stay unique if called multiple times
        offset = len(self._chunks)
        for c in new_chunks:
            c.chunk_id += offset
        self._chunks.extend(new_chunks)

        chunk_texts = [c.text for c in self._chunks]
        self.embedding_model.fit(chunk_texts)  # re-fit TF-IDF backend on full corpus
        all_embeddings = self.embedding_model.encode(chunk_texts)

        # Rebuild vector store from scratch since TF-IDF vocab may have changed
        self.vector_store = VectorStore()
        self.vector_store.add(all_embeddings, self._chunks)
        self._is_built = True

    # ------------------------------------------------------------------
    def ask(self, question: str, top_k: int = None) -> RAGAnswer:
        """Stages 5-7: embed the query, retrieve top chunks, generate an answer."""
        if not self._is_built:
            raise RuntimeError("No documents ingested yet. Call .ingest([...]) first.")

        top_k = top_k or self.top_k

        # Stage 5: Query Processing
        query_embedding = self.embedding_model.encode([question])[0]

        # Stage 6: Context Retrieval
        results = self.vector_store.search(query_embedding, top_k=top_k)

        # Stage 7: Answer Generation
        answer = self.generator.generate(question, results)

        return RAGAnswer(question=question, answer=answer, sources=results)

    def info(self) -> str:
        return (
            f"Embedding backend : {self.embedding_model.backend_name}\n"
            f"Generation backend: {self.generator.backend_name}\n"
            f"Chunks indexed    : {len(self._chunks)}"
        )
