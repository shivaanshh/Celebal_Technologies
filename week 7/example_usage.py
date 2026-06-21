"""
example_usage.py
-----------------
A minimal scripted demo of using RAGPipeline directly from Python,
without the CLI. Run with:

    python example_usage.py
"""

from src.pipeline import RAGPipeline


def main():
    pipeline = RAGPipeline(chunk_size=150, chunk_overlap=30, top_k=3)
    pipeline.ingest(["data/sample_document.txt"])

    print(pipeline.info())
    print()

    questions = [
        "What is Retrieval-Augmented Generation?",
        "Why might a system use a vector database instead of a linear scan?",
        "What are some ways to improve a RAG system's retrieval quality?",
    ]

    for q in questions:
        result = pipeline.ask(q)
        print(f"Q: {q}")
        print(f"A: {result.answer}")
        print(f"   (top source score: {result.sources[0].score:.3f})" if result.sources else "")
        print()


if __name__ == "__main__":
    main()
