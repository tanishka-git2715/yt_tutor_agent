"""
STEP 2 — Chunk transcripts and build the vector database
---------------------------------------------------------
Usage:
    python scripts/2_build_vectorstore.py

Input:
    data/transcripts_raw.json

Output:
    vectorstore/   — ChromaDB persistent database (free, local, no API key)
    data/chunks_preview.json  — first 20 chunks for inspection

How it works:
    1. Split each transcript into overlapping 400-word chunks
    2. Embed each chunk using sentence-transformers (free, runs locally)
    3. Store in ChromaDB with metadata: video_id, title, url, chunk_index
"""

import json
import os
import sys
import re

INPUT_FILE = "data/transcripts_raw.json"
VECTORSTORE_DIR = "vectorstore"
CHUNK_SIZE = 400        # words per chunk
CHUNK_OVERLAP = 60      # words of overlap between chunks
EMBED_MODEL = "all-MiniLM-L6-v2"   # fast, free, runs locally — 384 dimensions


def install_deps():
    pkgs = ["chromadb", "huggingface_hub"]
    for p in pkgs:
        try:
            __import__(p.replace("-", "_"))
        except ImportError:
            print(f"Installing {p}...")
            os.system(f"{sys.executable} -m pip install {p} -q")

install_deps()

import chromadb  # noqa
from huggingface_hub import InferenceClient  # noqa


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-window chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def build_vectorstore():
    # Load transcripts
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found. Run step 1 first.")
        sys.exit(1)

    with open(INPUT_FILE) as f:
        transcripts = json.load(f)

    print(f"Loaded {len(transcripts)} transcripts")

    # Define API embedding function
    print(f"\nUsing HuggingFace Inference API: {EMBED_MODEL}")
    client = InferenceClient(token=os.environ.get("HF_TOKEN"))
    
    def embedding_fn(texts: list[str]) -> list[list[float]]:
        try:
            embeddings = client.feature_extraction(
                texts, 
                model=f"sentence-transformers/{EMBED_MODEL}"
            )
            # Ensure it is a list of lists
            if isinstance(embeddings, list) and not isinstance(embeddings[0], list):
                return [embeddings]
            return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
        except Exception as e:
            if "503" in str(e):
                import time
                print("  Model is loading on HF... waiting 20s")
                time.sleep(20)
                return embedding_fn(texts)
            raise e
    
    print("API setup complete.")

    # Set up ChromaDB
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)

    # Delete existing collection if rebuilding
    try:
        client.delete_collection("yt_transcripts")
        print("Deleted existing collection (rebuilding)")
    except Exception:
        pass

    collection = client.create_collection(
        name="yt_transcripts",
        metadata={"hnsw:space": "cosine"},
    )

    # Process each transcript
    all_chunks = []
    all_ids = []
    all_metadatas = []
    preview_chunks = []

    total_chunks = 0
    for i, item in enumerate(transcripts):
        chunks = chunk_text(item["transcript"])
        print(f"[{i+1}/{len(transcripts)}] {item['title'][:60]} -> {len(chunks)} chunks")

        for j, chunk in enumerate(chunks):
            chunk_id = f"{item['video_id']}__chunk{j}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append({
                "video_id": item["video_id"],
                "title": item["title"],
                "url": item["url"],
                "chunk_index": j,
                "total_chunks": len(chunks),
            })
            total_chunks += 1

            if len(preview_chunks) < 20:
                preview_chunks.append({"id": chunk_id, "title": item["title"], "text": chunk[:300]})

    print(f"\nTotal chunks to embed: {total_chunks}")
    print("Embedding all chunks... (this may take a few minutes)")

    # Embed in batches of 32 (smaller for API)
    BATCH = 32
    all_embeddings = []
    for start in range(0, len(all_chunks), BATCH):
        batch = all_chunks[start : start + BATCH]
        embeddings = embedding_fn(batch)
        all_embeddings.extend(embeddings)
        done = min(start + BATCH, len(all_chunks))
        pct = done / len(all_chunks) * 100
        print(f"  Embedded {done}/{len(all_chunks)} ({pct:.0f}%)", end="\r")

    print("\nStoring in ChromaDB...")

    # Insert in batches (ChromaDB has insert limits)
    for start in range(0, len(all_chunks), BATCH):
        collection.add(
            documents=all_chunks[start : start + BATCH],
            embeddings=all_embeddings[start : start + BATCH],
            ids=all_ids[start : start + BATCH],
            metadatas=all_metadatas[start : start + BATCH],
        )

    # Save preview
    with open("data/chunks_preview.json", "w") as f:
        json.dump(preview_chunks, f, indent=2, ensure_ascii=False)

    # Save metadata file for the app to read
    meta = {
        "total_videos": len(transcripts),
        "total_chunks": total_chunks,
        "embed_model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "videos": [{"video_id": t["video_id"], "title": t["title"], "url": t["url"]} for t in transcripts],
    }
    with open("data/vectorstore_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDONE! Vector database built.")
    print(f"  {len(transcripts)} videos  |  {total_chunks} chunks  ->  vectorstore/")
    print(f"  Preview saved → data/chunks_preview.json")


if __name__ == "__main__":
    build_vectorstore()
