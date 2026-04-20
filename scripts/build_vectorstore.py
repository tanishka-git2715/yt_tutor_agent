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
INDEX_DIR = "data/search_index"
INDEX_FILE = "data/search_index/tfidf_index.joblib"
CHUNK_SIZE = 400        # words per chunk
CHUNK_OVERLAP = 60      # words of overlap between chunks


def install_deps():
    pkgs = ["scikit-learn", "joblib"]
    for p in pkgs:
        try:
            __import__(p.replace("-", "_"))
        except ImportError:
            print(f"Installing {p}...")
            os.system(f'"{sys.executable}" -m pip install {p} -q')

install_deps()

from sklearn.feature_extraction.text import TfidfVectorizer # noqa
import joblib # noqa


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


def build_vectorstore(progress_cb=None):
    # Load transcripts
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found. Run step 1 first.")
        sys.exit(1)

    with open(INPUT_FILE) as f:
        transcripts = json.load(f)

    print(f"Loaded {len(transcripts)} transcripts")

    # Setup indexing
    print(f"\nBuilding TF-IDF Search Index...")
    vectorizer = TfidfVectorizer(stop_words='english', min_df=2)

    # Set up index directory
    os.makedirs(INDEX_DIR, exist_ok=True)

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

    print(f"\nTotal chunks to index: {total_chunks}")
    print("Fitting TF-IDF and building matrix...")

    # Fit TF-IDF on all chunks
    if progress_cb: progress_cb(0.5, "Fitting TF-IDF...")
    tfidf_matrix = vectorizer.fit_transform(all_chunks)

    if progress_cb: progress_cb(0.8, "Saving index...")
    print("\nSaving index...")

    # Save to joblib
    index_data = {
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "chunks": [
            {
                "video_id": m["video_id"],
                "title": m["title"],
                "url": m["url"],
                "chunk_index": m["chunk_index"],
                "text": c
            }
            for m, c in zip(all_metadatas, all_chunks)
        ]
    }
    joblib.dump(index_data, INDEX_FILE)

    # Save preview
    with open("data/chunks_preview.json", "w") as f:
        json.dump(preview_chunks, f, indent=2, ensure_ascii=False)

    # Save metadata file for the app to read
    meta = {
        "total_videos": len(transcripts),
        "total_chunks": total_chunks,
        "method": "tfidf",
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "videos": [{"video_id": t["video_id"], "title": t["title"], "url": t["url"]} for t in transcripts],
    }
    with open("data/vectorstore_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDONE! Search index built.")
    print(f"  {len(transcripts)} videos  |  {total_chunks} chunks  ->  {INDEX_DIR}/")
    print(f"  Preview saved -> data/chunks_preview.json")


if __name__ == "__main__":
    build_vectorstore()
