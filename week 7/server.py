import os
import shutil
import tempfile
import threading
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.document_loader import load_document, DocumentLoadError
from src.chunking import chunk_documents
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.generator import AnswerGenerator
from src.pipeline import RAGPipeline

app = FastAPI(title="RAG QA", docs_url=None, redoc_url=None)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_pipeline: Optional[RAGPipeline] = None
_loaded_files: list[str] = []

# Embedding model cached across ingestions — avoids reloading sentence-transformers
_embed_model: Optional[EmbeddingModel] = None
_embed_lock = threading.Lock()

# Live ingestion progress
_progress = {"status": "idle", "stage": "", "error": None, "result": None}
_progress_lock = threading.Lock()


def _get_embed_model() -> EmbeddingModel:
    global _embed_model
    with _embed_lock:
        if _embed_model is None:
            _embed_model = EmbeddingModel()
        return _embed_model


def _set_stage(stage: str):
    with _progress_lock:
        _progress["stage"] = stage


# ---------------------------------------------------------------------------
# Background ingestion (runs in a thread so the HTTP response returns fast)
# ---------------------------------------------------------------------------
def _run_ingest(tmp_dir: str, tmp_paths: list, backend: str, model_name: str):
    global _pipeline, _loaded_files
    try:
        # Stage 1 — load docs (individually to capture per-file errors)
        _set_stage("Loading documents…")
        docs = []
        skipped = []          # list of {"name": str, "reason": str}
        for tmp_path, orig_name in tmp_paths:
            try:
                docs.append(load_document(tmp_path))
            except DocumentLoadError as exc:
                skipped.append({"name": orig_name, "reason": str(exc)})
                print(f"[warning] Skipping '{orig_name}': {exc}")

        if not docs:
            reasons = "; ".join(f"{s['name']}: {s['reason']}" for s in skipped)
            raise RuntimeError(
                f"No text could be extracted from the uploaded file(s). "
                f"Details: {reasons}"
            )

        # Stage 2 — chunk
        _set_stage(f"Chunking {len(docs)} document(s)…")
        chunks = chunk_documents(docs, chunk_size=150, overlap=30)

        # Stage 3 — embed (slow on first call; cached after)
        _set_stage(f"Embedding {len(chunks)} chunks (first run loads model)…")
        embed = _get_embed_model()
        embed.fit([c.text for c in chunks])
        embeddings = embed.encode([c.text for c in chunks])

        # Stage 4 — index
        _set_stage("Building vector index…")
        store = VectorStore()
        store.add(embeddings, chunks)

        # Stage 5 — wire up generator
        _set_stage("Connecting LLM…")
        generator = AnswerGenerator(backend=backend, model=model_name or None)

        # Assemble pipeline object so .ask() works
        p = RAGPipeline.__new__(RAGPipeline)
        p.chunk_size      = 150
        p.chunk_overlap   = 30
        p.top_k           = 3
        p.embedding_model = embed
        p.vector_store    = store
        p.generator       = generator
        p._chunks         = chunks
        p._is_built       = True

        _pipeline    = p
        _loaded_files = [os.path.basename(d.source_path) for d in docs]

        from collections import Counter
        source_counts = Counter(os.path.basename(c.source) for c in chunks)
        doc_stats = [{"name": n, "chunks": ct} for n, ct in source_counts.items()]

        with _progress_lock:
            _progress.update({
                "status": "done",
                "stage":  "Ready",
                "error":  None,
                "result": {
                    "files":               _loaded_files,
                    "skipped":             skipped,
                    "chunks":              len(chunks),
                    "doc_stats":           doc_stats,
                    "generation_backend":  generator.backend_name,
                    "model":               generator._model,
                    "embedding_backend":   embed.backend_name,
                },
            })
    except Exception as exc:
        with _progress_lock:
            _progress.update({"status": "error", "stage": "", "error": str(exc), "result": None})
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    history: list[dict] = []


class ConfigRequest(BaseModel):
    groq_api_key:      str = ""
    anthropic_api_key: str = ""
    openai_api_key:    str = ""


@app.get("/api/status")
async def api_status():
    if not _pipeline or not _pipeline._is_built:
        return {"ready": False, "files": [], "chunks": 0, "doc_stats": [],
                "embedding_backend": None, "generation_backend": None, "model": None}
    from collections import Counter
    source_counts = Counter(os.path.basename(c.source) for c in _pipeline._chunks)
    doc_stats = [{"name": n, "chunks": ct} for n, ct in source_counts.items()]
    return {
        "ready":              True,
        "files":              _loaded_files,
        "chunks":             len(_pipeline._chunks),
        "doc_stats":          doc_stats,
        "embedding_backend":  _pipeline.embedding_model.backend_name,
        "generation_backend": _pipeline.generator.backend_name,
        "model":              _pipeline.generator._model,
    }


@app.get("/api/ingest-progress")
async def api_ingest_progress():
    with _progress_lock:
        return dict(_progress)


@app.post("/api/config")
async def api_config(req: ConfigRequest):
    if req.groq_api_key:      os.environ["GROQ_API_KEY"]      = req.groq_api_key
    if req.anthropic_api_key: os.environ["ANTHROPIC_API_KEY"] = req.anthropic_api_key
    if req.openai_api_key:    os.environ["OPENAI_API_KEY"]    = req.openai_api_key
    return {"status": "ok"}


@app.post("/api/ingest")
async def api_ingest(
    files:   list[UploadFile] = File(...),
    backend: str = Form("auto"),
    model:   str = Form(""),
):
    # Save uploads to a temp dir (preserved until the thread cleans up)
    tmp_dir = tempfile.mkdtemp()
    tmp_paths: list[tuple[str, str]] = []
    for f in files:
        dest = os.path.join(tmp_dir, f.filename)
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        tmp_paths.append((dest, f.filename))

    with _progress_lock:
        _progress.update({"status": "running", "stage": "Starting…", "error": None, "result": None})

    threading.Thread(
        target=_run_ingest,
        args=(tmp_dir, tmp_paths, backend, model.strip()),
        daemon=True,
    ).start()

    return {"status": "started"}


@app.post("/api/ask")
async def api_ask(req: AskRequest):
    if not _pipeline or not _pipeline._is_built:
        raise HTTPException(400, "No documents loaded. Upload files first.")

    # Expand short/vague follow-up queries with prior context for better retrieval
    search_query = req.question
    if len(req.question.split()) <= 4 and req.history:
        prev = next((h["content"] for h in reversed(req.history) if h["role"] == "user"), "")
        if prev:
            search_query = f"{prev} {req.question}"

    result = _pipeline.ask(search_query, top_k=req.top_k)

    # Generate with the original question + history so phrasing stays natural
    answer = _pipeline.generator.generate(
        req.question, result.sources, history=req.history
    )

    return {
        "answer": answer,
        "sources": [
            {"chunk_id": s.chunk_id, "text": s.text,
             "source": os.path.basename(s.source), "score": round(s.score, 3)}
            for s in result.sources
        ],
    }


@app.get("/api/ollama-models")
async def ollama_models():
    import requests as req
    try:
        resp = req.get("http://localhost:11434/api/tags", timeout=3)
        resp.raise_for_status()
        return {"models": [m["name"] for m in resp.json().get("models", [])]}
    except Exception:
        return {"models": []}


@app.post("/api/reset")
async def api_reset():
    global _pipeline, _loaded_files
    _pipeline     = None
    _loaded_files = []
    with _progress_lock:
        _progress.update({"status": "idle", "stage": "", "error": None, "result": None})
    return {"status": "reset"}


# ---------------------------------------------------------------------------
# Static + SPA
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")
