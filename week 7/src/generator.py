"""
generator.py
------------
Stage 7 of the RAG pipeline: Answer Generation.

Takes the user's question plus the retrieved context chunks (the
"augmentation" step) and produces a final, grounded answer using a
language model.

Five backends are supported, auto-selected in this order:

1. AnthropicBackend  - Claude via Anthropic API (ANTHROPIC_API_KEY env var)
2. GroqBackend       - Fast cloud inference via Groq (GROQ_API_KEY env var)
3. OllamaBackend     - Local models via Ollama (must be running on localhost)
4. OpenAIBackend     - GPT via OpenAI API (OPENAI_API_KEY env var)
5. LocalExtractiveBackend - no API key / no internet required.
"""

from __future__ import annotations

import os
import re
from typing import List

from .vector_store import SearchResult


SYSTEM_PROMPT = (
    "You are a knowledgeable document assistant. "
    "Answer the user's question using the provided context excerpts as your primary source. "
    "If a question is a follow-up (e.g. 'explain', 'elaborate', 'go deeper'), use the conversation "
    "history to understand what to explain, then draw on the context excerpts for detail. "
    "Give thorough, well-structured answers. Cite which excerpt(s) you used by their [number]. "
    "If the context genuinely lacks the information needed, say so briefly — but first try to "
    "synthesise what IS available rather than refusing outright."
)


def _build_context_block(results: List[SearchResult]) -> str:
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(f"[{i}] (source: {os.path.basename(r.source)})\n{r.text}")
    return "\n\n".join(blocks)


def _build_user_prompt(question: str, results: List[SearchResult]) -> str:
    context_block = _build_context_block(results)
    return (
        f"Context excerpts from the document(s):\n{context_block}\n\n"
        f"Question: {question}"
    )


_DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "mistral:latest",
    "openai": "gpt-4o-mini",
}

_OLLAMA_HOST = "http://localhost:11434"


class AnswerGenerator:
    def __init__(self, backend: str = "auto", model: str = None):
        """
        Args:
            backend: one of "auto", "anthropic", "groq", "ollama", "openai", "local".
                     "auto" picks the best available option.
            model: override the default model for the chosen backend.
        """
        self.backend_name: str
        self._client = None
        self._model: str = model  # resolved once backend is decided

        if backend in ("auto", "anthropic") and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import anthropic

                self._client = anthropic.Anthropic()
                self.backend_name = "anthropic"
                self._model = model or _DEFAULT_MODELS["anthropic"]
                return
            except Exception as exc:
                if backend == "anthropic":
                    raise
                print(f"[info] Anthropic backend unavailable: {exc}")

        if backend in ("auto", "groq") and os.environ.get("GROQ_API_KEY"):
            try:
                from groq import Groq

                self._client = Groq(api_key=os.environ["GROQ_API_KEY"])
                self.backend_name = "groq"
                self._model = model or _DEFAULT_MODELS["groq"]
                return
            except Exception as exc:
                if backend == "groq":
                    raise
                print(f"[info] Groq backend unavailable: {exc}")

        if backend in ("auto", "ollama"):
            try:
                import requests

                resp = requests.get(f"{_OLLAMA_HOST}/api/tags", timeout=3)
                resp.raise_for_status()
                self.backend_name = "ollama"
                self._model = model or _DEFAULT_MODELS["ollama"]
                return
            except Exception as exc:
                if backend == "ollama":
                    raise RuntimeError(
                        f"Ollama is not reachable at {_OLLAMA_HOST}. "
                        "Make sure Ollama is running (`ollama serve`) and the model is pulled "
                        f"(`ollama pull {model or _DEFAULT_MODELS['ollama']}`)."
                    ) from exc
                print(f"[info] Ollama backend unavailable ({exc.__class__.__name__}). "
                      "Start Ollama with `ollama serve` to enable it.")

        if backend in ("auto", "openai") and os.environ.get("OPENAI_API_KEY"):
            try:
                import openai

                self._client = openai.OpenAI()
                self.backend_name = "openai"
                self._model = model or _DEFAULT_MODELS["openai"]
                return
            except Exception as exc:
                if backend == "openai":
                    raise
                print(f"[info] OpenAI backend unavailable: {exc}")

        if backend in ("anthropic", "groq", "openai"):
            raise RuntimeError(
                f"Requested backend '{backend}' but no valid API key/client was found."
            )

        self.backend_name = "local"

    # ------------------------------------------------------------------
    def generate(
        self,
        question: str,
        results: List[SearchResult],
        max_tokens: int = 800,
        history: List[dict] = None,
    ) -> str:
        if not results:
            return "I couldn't find any relevant information in the document(s) to answer that question."

        history = history or []
        if self.backend_name == "anthropic":
            return self._generate_anthropic(question, results, max_tokens, history)
        if self.backend_name == "groq":
            return self._generate_groq(question, results, max_tokens, history)
        if self.backend_name == "ollama":
            return self._generate_ollama(question, results, max_tokens, history)
        if self.backend_name == "openai":
            return self._generate_openai(question, results, max_tokens, history)
        return self._generate_local(question, results)

    # ------------------------------------------------------------------
    def _chat_messages(self, question: str, results: List[SearchResult], history: List[dict]) -> List[dict]:
        """Build the messages list: optional prior turns + current question with context."""
        msgs = list(history[-6:])  # keep last 3 exchanges (6 messages)
        msgs.append({"role": "user", "content": _build_user_prompt(question, results)})
        return msgs

    def _generate_anthropic(self, question, results, max_tokens, history) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=self._chat_messages(question, results, history),
        )
        return "".join(block.text for block in message.content if block.type == "text").strip()

    def _generate_groq(self, question, results, max_tokens, history) -> str:
        completion = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}]
                     + self._chat_messages(question, results, history),
        )
        return completion.choices[0].message.content.strip()

    def _generate_ollama(self, question, results, max_tokens, history) -> str:
        import requests

        payload = {
            "model": self._model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}]
                        + self._chat_messages(question, results, history),
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        resp = requests.post(f"{_OLLAMA_HOST}/api/chat", json=payload, timeout=120)
        if resp.status_code == 404:
            raise RuntimeError(
                f"Ollama model '{self._model}' not found. "
                f"Pull it first with: ollama pull {self._model}"
            )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def _generate_openai(self, question, results, max_tokens, history) -> str:
        completion = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}]
                     + self._chat_messages(question, results, history),
        )
        return completion.choices[0].message.content.strip()

    def _generate_local(self, question: str, results: List[SearchResult]) -> str:
        """No-LLM fallback: extractive summary of the best-matching sentences.

        This does not "understand" the question the way an LLM would — it
        ranks sentences from the retrieved chunks by lexical overlap with the
        question and stitches together the most relevant ones. It exists so
        the full pipeline (retrieve -> augment -> generate) is runnable
        without any API key, for demos and offline use.
        """
        question_words = set(re.findall(r"[a-zA-Z]+", question.lower()))
        question_words = {w for w in question_words if len(w) > 2}

        scored_sentences = []
        for r in results:
            sentences = re.split(r"(?<=[.!?])\s+", r.text)
            for sent in sentences:
                sent_words = set(re.findall(r"[a-zA-Z]+", sent.lower()))
                overlap = len(question_words & sent_words)
                if sent.strip():
                    scored_sentences.append((overlap, r.score, sent.strip(), r.source))

        scored_sentences.sort(key=lambda x: (x[0], x[1]), reverse=True)
        top_sentences = scored_sentences[:3] if scored_sentences else []

        if not top_sentences or all(s[0] == 0 for s in top_sentences):
            # No direct lexical overlap found; just summarize the top chunk.
            top = results[0]
            answer = top.text[:400] + ("..." if len(top.text) > 400 else "")
            return (
                f"[local fallback mode — no LLM API key configured]\n\n"
                f"Based on the most relevant retrieved excerpt (source: {os.path.basename(top.source)}):\n"
                f"{answer}"
            )

        bullet_lines = "\n".join(f"- {s[2]} (source: {os.path.basename(s[3])})" for s in top_sentences)
        return (
            "[local fallback mode — no LLM API key configured, showing extracted excerpts]\n\n"
            f"{bullet_lines}\n\n"
            "Tip: set ANTHROPIC_API_KEY or OPENAI_API_KEY to get a fully synthesized, "
            "natural-language answer instead of raw excerpts."
        )
