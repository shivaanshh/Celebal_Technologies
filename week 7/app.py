#!/usr/bin/env python3
"""
app.py
------
Command-line interface for the RAG Document Question-Answering system.

Usage:
    # Interactive mode: ingest documents, then ask questions in a loop
    python app.py --docs data/sample_document.txt

    # One-shot mode: ask a single question and exit
    python app.py --docs notes.pdf resume.pdf --question "What is the candidate's experience?"

Set ANTHROPIC_API_KEY or OPENAI_API_KEY as an environment variable to enable
real LLM-generated answers. Without a key, the system runs in a local,
extractive fallback mode so the full pipeline still works end-to-end.
"""

import argparse
import sys

from src.pipeline import RAGPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="RAG Document Question-Answering System")
    parser.add_argument("--docs", nargs="+", required=True, help="Path(s) to document file(s) (.txt, .md, .pdf)")
    parser.add_argument("--question", "-q", default=None, help="Ask a single question and exit (non-interactive)")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve per query (default: 3)")
    parser.add_argument("--chunk-size", type=int, default=150, help="Words per chunk (default: 150)")
    parser.add_argument("--chunk-overlap", type=int, default=30, help="Word overlap between chunks (default: 30)")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "anthropic", "groq", "ollama", "openai", "local"],
        help="LLM backend (default: auto). groq needs GROQ_API_KEY; ollama needs `ollama serve`.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name to use with the chosen backend. "
            "Defaults: groq=llama-3.3-70b-versatile, ollama=llama3.2, "
            "anthropic=claude-haiku-4-5-20251001, openai=gpt-4o-mini"
        ),
    )
    return parser.parse_args()


def print_answer(result):
    print("\n" + "=" * 70)
    print(f"Q: {result.question}")
    print("-" * 70)
    print(result.answer)
    print("-" * 70)
    print("Sources used:")
    for i, s in enumerate(result.sources, start=1):
        preview = s.text[:100].replace("\n", " ") + ("..." if len(s.text) > 100 else "")
        print(f"  [{i}] score={s.score:.3f}  ({s.source})  \"{preview}\"")
    print("=" * 70 + "\n")


def main():
    args = parse_args()

    pipeline = RAGPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        llm_backend=args.backend,
        llm_model=args.model,
    )

    try:
        pipeline.ingest(args.docs)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(pipeline.info())

    if args.question:
        result = pipeline.ask(args.question)
        print_answer(result)
        return

    print("\nEnter your questions below (type 'exit' or 'quit' to stop).")
    while True:
        try:
            question = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        result = pipeline.ask(question)
        print_answer(result)


if __name__ == "__main__":
    main()
