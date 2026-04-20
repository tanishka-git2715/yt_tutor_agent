"""
rag_engine.py — Core retrieval + Claude API logic
---------------------------------------------------
This module is imported by the Streamlit app and the CLI.
It handles:
  - Loading the vector database
  - Embedding user questions
  - Retrieving the most relevant transcript chunks
  - Building the prompt and calling Claude
"""

import json
import os
import sys
from dataclasses import dataclass
from typing import Generator
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VECTORSTORE_DIR = "data/search_index"
META_FILE = "data/vectorstore_meta.json"
INDEX_FILE = "data/search_index/tfidf_index.joblib"
TOP_K = 6               # number of chunks to retrieve
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"
MAX_TOKENS = 1500

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SourceChunk:
    video_id: str
    title: str
    url: str
    chunk_index: int
    text: str
    score: float


@dataclass
class TutorResponse:
    answer: str
    sources: list[SourceChunk]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class YTTutorEngine:
    def __init__(self, creator_name: str = "the creator", creator_style: str = ""):
        self.creator_name = creator_name
        self.creator_style = creator_style
        self._collection = None
        self._embed_model = None
        self._meta = None
        self._anthropic = None
        self._groq = None
        self.provider = "anthropic" # default

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _load_meta(self):
        if self._meta is None:
            if not os.path.exists(META_FILE):
                raise FileNotFoundError(
                    f"Metadata file not found: {META_FILE}\n"
                    "Run scripts/2_build_vectorstore.py first."
                )
            with open(META_FILE, encoding="utf-8") as f:
                self._meta = json.load(f)
        return self._meta

    def _load_search_index(self):
        if self._collection is None:
            import joblib
            if not os.path.exists(INDEX_FILE):
                raise FileNotFoundError(
                    f"Search index not found at: {INDEX_FILE}\n"
                    "Run scripts/2_build_vectorstore.py first."
                )
            self._collection = joblib.load(INDEX_FILE) # This will be our dict {vectorizer, matrix, chunks}
        return self._collection

    def _load_anthropic(self, api_key: str | None = None):
        if self._anthropic is None:
            import anthropic
            key = api_key or ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY", "")
            if not key:
                raise ValueError("ANTHROPIC_API_KEY not set.")
            self._anthropic = anthropic.Anthropic(api_key=key)
        return self._anthropic

    def _load_groq(self, api_key: str | None = None):
        if self._groq is None:
            from groq import Groq
            key = api_key or GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
            if not key:
                raise ValueError("GROQ_API_KEY not set.")
            self._groq = Groq(api_key=key)
        return self._groq

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def meta(self) -> dict:
        return self._load_meta()

    def is_ready(self) -> bool:
        try:
            self._load_meta()
            self._load_search_index()
            return True
        except Exception:
            return False

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[SourceChunk]:
        """Find the most relevant transcript chunks using TF-IDF."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        index = self._load_search_index()
        vectorizer = index["vectorizer"]
        tfidf_matrix = index["tfidf_matrix"]
        chunk_data = index["chunks"]
        
        # Vectorize user question
        query_vec = vectorizer.transform([question])
        
        # Compute similarities
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # Get top K indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        chunks = []
        for idx in top_indices:
            score = similarities[idx]
            if score < 0.01: continue
            
            meta = chunk_data[idx]
            chunks.append(SourceChunk(
                video_id=meta["video_id"],
                title=meta["title"],
                url=meta["url"],
                chunk_index=meta["chunk_index"],
                text=meta["text"],
                score=float(score),
            ))
        return chunks

    def build_system_prompt(self) -> str:
        style_line = (
            f"\n\nTeaching style: {self.creator_style}"
            if self.creator_style else ""
        )
        meta = self._load_meta()
        return f"""You are an AI tutor that embodies the knowledge and teaching style of {self.creator_name}.

You have been trained exclusively on the transcripts from {self.creator_name}'s YouTube channel, which contains {meta['total_videos']} videos.

Your job:
- Answer questions using ONLY the knowledge present in the transcript excerpts provided in each message.
- Teach in the spirit and style of {self.creator_name} — use their vocabulary, examples, and way of explaining things where possible.
- If a concept is covered across multiple videos, synthesise it naturally.
- Always cite which video(s) your answer comes from (use the title, not the URL).
- If the question cannot be answered from the provided excerpts, say so clearly and suggest the user rephrase or ask something that {self.creator_name} has covered.
- Never make up information. Never draw on knowledge outside the provided transcripts.
- Be conversational, helpful, and encouraging — like a personal tutor who has watched every video.{style_line}

Format:
- Give clear, structured answers.
- End with: "Source: [Video Title(s)]" so the user knows where the knowledge comes from.
- For complex topics, break answers into numbered steps or bullet points."""

    def ask(self, question: str, chat_history: list[dict] | None = None, provider: str = "anthropic", api_key: str | None = None) -> TutorResponse:
        """Single-turn or multi-turn question. Returns full response."""
        chunks = self.retrieve(question)
        context = self._format_context(chunks)
        messages = self._build_messages(question, context, chat_history)
        system_prompt = self.build_system_prompt()

        if provider == "groq":
            client = self._load_groq(api_key)
            response = client.chat.completions.create(
                model=DEFAULT_GROQ_MODEL,
                messages=[{"role": "system", "content": system_prompt}] + messages,
                max_tokens=MAX_TOKENS,
            )
            answer = response.choices[0].message.content
        else:
            client = self._load_anthropic(api_key)
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                messages=messages,
            )
            answer = response.content[0].text
            
        return TutorResponse(answer=answer, sources=chunks)

    def ask_stream(
        self, question: str, chat_history: list[dict] | None = None, provider: str = "anthropic", api_key: str | None = None
    ) -> tuple[list[SourceChunk], Generator]:
        """Streaming version — yields text tokens. Returns (chunks, generator)."""
        chunks = self.retrieve(question)
        context = self._format_context(chunks)
        messages = self._build_messages(question, context, chat_history)
        system_prompt = self.build_system_prompt()

        if provider == "groq":
            client = self._load_groq(api_key)
            # Create a wrapper generator to match Claude's stream structure if needed
            # or just handle the tokens in app.py. 
            # For simplicity, we'll return the raw groq stream and handle it.
            stream = client.chat.completions.create(
                model=DEFAULT_GROQ_MODEL,
                messages=[{"role": "system", "content": system_prompt}] + messages,
                max_tokens=MAX_TOKENS,
                stream=True,
            )
            
            def groq_generator():
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            
            return chunks, groq_generator()
        else:
            client = self._load_anthropic(api_key)
            stream_ctx = client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                messages=messages,
            )
            
            def anthropic_generator():
                with stream_ctx as s:
                    for text in s.text_stream:
                        yield text
            
            return chunks, anthropic_generator()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_context(self, chunks: list[SourceChunk]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(
                f"[Excerpt {i} — from \"{c.title}\" ({c.url})]\n{c.text}"
            )
        return "\n\n---\n\n".join(parts)

    def _build_messages(
        self,
        question: str,
        context: str,
        chat_history: list[dict] | None,
    ) -> list[dict]:
        history = chat_history or []

        user_message = (
            f"TRANSCRIPT EXCERPTS (use only these to answer):\n\n"
            f"{context}\n\n"
            f"---\n\n"
            f"MY QUESTION: {question}"
        )

        messages = []
        for turn in history[-6:]:   # keep last 3 exchanges for context
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_message})
        return messages
