# RAG Document Question-Answering System

A simple, from-scratch Retrieval-Augmented Generation (RAG) pipeline for asking
questions about your own documents (PDF, TXT, MD). Built to clearly demonstrate
every stage of a RAG system rather than hiding it behind a single framework call.

## How it works

```
   Your Documents (PDF/TXT)
            │
            ▼
   1. Document Ingestion      (src/document_loader.py)
            │
            ▼
   2. Text Chunking           (src/chunking.py)
            │
            ▼
   3. Embedding Creation      (src/embeddings.py)
            │
            ▼
   4. Vector Store            (src/vector_store.py)
            │
   ┌────────┴─────────┐
   │   User Question   │
   └────────┬─────────┘
            ▼
   5. Query Embedding         (src/embeddings.py)
            │
            ▼
   6. Context Retrieval       (src/vector_store.py — cosine similarity search)
            │
            ▼
   7. Answer Generation       (src/generator.py — LLM call with retrieved context)
            │
            ▼
       Grounded Answer
```

## Quick start

```bash
pip install -r requirements.txt

# Ask one question
python app.py --docs data/sample_document.txt --question "What is RAG?"

# Or chat interactively over your own files
python app.py --docs my_resume.pdf my_notes.txt
```

## LLM backends

Five backends are supported, auto-detected in this order:

| Backend | How to enable | Default model |
|---------|---------------|---------------|
| `anthropic` | `$env:ANTHROPIC_API_KEY="sk-ant-..."` | claude-haiku-4-5-20251001 |
| `groq` | `$env:GROQ_API_KEY="gsk_..."` | llama-3.3-70b-versatile |
| `ollama` | Run `ollama serve` locally | mistral:latest |
| `openai` | `$env:OPENAI_API_KEY="sk-..."` | gpt-4o-mini |
| `local` | always available, no key needed | extractive fallback |

**Ollama (fully local, no API key):**
```powershell
# install from https://ollama.com, then:
ollama pull mistral
python app.py --docs data/sample_document.txt --backend ollama
# use a different local model:
python app.py --docs notes.pdf --backend ollama --model phi3:latest
```

**Groq (fastest cloud inference):**
```powershell
$env:GROQ_API_KEY = "gsk_..."
python app.py --docs data/sample_document.txt --backend groq
# use a different Groq model:
python app.py --docs notes.pdf --backend groq --model mixtral-8x7b-32768
```

Without any key or running Ollama instance, the system falls back to **local extractive mode** — retrieval still works with real semantic search, answers are extracted sentences rather than synthesized text.

## Using better embeddings

By default, if the `sentence-transformers` package and internet access to
Hugging Face are available, the system downloads `all-MiniLM-L6-v2` and uses
true semantic embeddings. If not, it automatically falls back to a TF-IDF +
SVD embedding built entirely with `scikit-learn` — no downloads required.
Either way, the rest of the pipeline (chunking, storage, retrieval,
generation) works identically; only `EmbeddingModel.backend_name` changes.

```bash
pip install sentence-transformers   # optional, for semantic embeddings
```

## Using it as a library

```python
from src.pipeline import RAGPipeline

pipeline = RAGPipeline(chunk_size=150, chunk_overlap=30, top_k=3)
pipeline.ingest(["my_document.pdf"])

result = pipeline.ask("What is the main idea of the document?")
print(result.answer)
for src in result.sources:
    print(src.source, src.score)
```

## Project structure

```
rag_qa_system/
├── app.py                  # CLI entry point
├── requirements.txt
├── data/
│   └── sample_document.txt # demo document about RAG itself
├── example_usage.py         # short scripted demo of the Python API
└── src/
    ├── document_loader.py   # Stage 1: load PDF/TXT/MD -> text
    ├── chunking.py           # Stage 2: text -> overlapping chunks
    ├── embeddings.py          # Stage 3/5: text -> vectors (ST or TF-IDF)
    ├── vector_store.py         # Stage 4/6: store + cosine similarity search
    ├── generator.py             # Stage 7: context + question -> answer
    └── pipeline.py               # orchestrates all stages
```

## Tuning parameters

| Flag              | Default | Effect                                                |
|--------------------|---------|--------------------------------------------------------|
| `--chunk-size`      | 150     | Words per chunk. Smaller = more precise retrieval, larger = more context per chunk. |
| `--chunk-overlap`   | 30      | Words shared between consecutive chunks, to avoid losing context at boundaries. |
| `--top-k`           | 3       | Number of chunks retrieved per question.               |
| `--backend`         | auto    | `anthropic`, `openai`, or `local` (extractive, no API). |

## Possible extensions (see project brief)

- Swap the in-memory vector store for FAISS/Chroma for larger corpora.
- Add hybrid search (BM25 keyword search + vector similarity).
- Add a re-ranking model over the initially retrieved chunks.
- Try different embedding models or chunking strategies (semantic chunking,
  recursive character splitting, etc).
