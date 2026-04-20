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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VECTORSTORE_DIR = "vectorstore"
META_FILE = "data/vectorstore_meta.json"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 8               # number of chunks to retrieve
CLAUDE_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1500

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


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
            with open(META_FILE) as f:
                self._meta = json.load(f)
        return self._meta

    def _load_collection(self):
        if self._collection is None:
            try:
                import chromadb
            except ImportError:
                os.system(f"{sys.executable} -m pip install chromadb -q")
                import chromadb

            if not os.path.exists(VECTORSTORE_DIR):
                raise FileNotFoundError(
                    f"Vector store not found at: {VECTORSTORE_DIR}\n"
                    "Run scripts/2_build_vectorstore.py first."
                )
            client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
            self._collection = client.get_collection("yt_transcripts")
        return self._collection

    def _load_embed_model(self):
        """Uses HuggingFace Inference API for embeddings."""
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=os.environ.get("HF_TOKEN"))
        
        def hf_embed(texts: list[str]) -> list[list[float]]:
            # InferenceClient.feature_extraction returns a numpy-like list of floats
            # for the given model. It handles partitioning/retries.
            embeddings = client.feature_extraction(
                texts, 
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            # Ensure it is a list of lists
            if isinstance(embeddings, list) and not isinstance(embeddings[0], list):
                return [embeddings]
            return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
            
        self._embed_model = hf_embed
        return self._embed_model

    def _load_anthropic(self):
        if self._anthropic is None:
            try:
                import anthropic
            except ImportError:
                os.system(f"{sys.executable} -m pip install anthropic -q")
                import anthropic
            key = ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY", "")
            if not key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not set.\n"
                    "Set it with:  export ANTHROPIC_API_KEY=sk-ant-..."
                )
            self._anthropic = anthropic.Anthropic(api_key=key)
        return self._anthropic

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def meta(self) -> dict:
        return self._load_meta()

    def is_ready(self) -> bool:
        try:
            self._load_meta()
            self._load_collection()
            return True
        except Exception:
            return False

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[SourceChunk]:
        """Find the most relevant transcript chunks for a question."""
        collection = self._load_collection()
        model = self._load_embed_model()
        
        # Call the HF API function we defined in _load_embed_model
        query_embedding = model([question])
        
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for doc, meta, dist in zip(docs, metas, dists):
            chunks.append(SourceChunk(
                video_id=meta["video_id"],
                title=meta["title"],
                url=meta["url"],
                chunk_index=meta["chunk_index"],
                text=doc,
                score=round(1 - dist, 3),   # cosine similarity
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

    def ask(self, question: str, chat_history: list[dict] | None = None) -> TutorResponse:
        """Single-turn or multi-turn question. Returns full response."""
        chunks = self.retrieve(question)
        context = self._format_context(chunks)
        messages = self._build_messages(question, context, chat_history)
        client = self._load_anthropic()

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            system=self.build_system_prompt(),
            messages=messages,
        )
        answer = response.content[0].text
        return TutorResponse(answer=answer, sources=chunks)

    def ask_stream(
        self, question: str, chat_history: list[dict] | None = None
    ) -> tuple[list[SourceChunk], Generator]:
        """Streaming version — yields text tokens. Returns (chunks, generator)."""
        chunks = self.retrieve(question)
        context = self._format_context(chunks)
        messages = self._build_messages(question, context, chat_history)
        client = self._load_anthropic()

        stream = client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            system=self.build_system_prompt(),
            messages=messages,
        )
        return chunks, stream

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
