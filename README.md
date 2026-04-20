# YouTube Channel AI Tutor

Turn any YouTube channel into a personal AI tutor powered by Claude.
Ask questions and get answers grounded exclusively in that creator's videos.

---

## How it works

1. **Extract** — Download transcripts from every video in a YouTube channel
2. **Index** — Chunk transcripts into 400-word pieces, embed them, store in a local vector database (ChromaDB, free, runs on your machine)
3. **Chat** — On each question, find the most relevant transcript chunks and send them to Claude with a tutor persona prompt

No fine-tuning. No cloud database. Runs entirely on your laptop after setup.

---

## Setup (one time)

### Prerequisites
- Python 3.10 or higher
- An Anthropic API key — get one at [console.anthropic.com](https://console.anthropic.com)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

The embedding model (`all-MiniLM-L6-v2`, ~90MB) downloads automatically on first run.

### 2. Extract transcripts from a YouTube channel

```bash
python scripts/1_extract_transcripts.py \
  --channel "https://www.youtube.com/@ChannelHandle/videos"
```

**Options:**
- `--limit 20` — only process the first 20 videos (good for testing)
- `--resume` — pick up where you left off if it was interrupted

**Output:** `data/transcripts_raw.json`

**Notes:**
- Works with any public YouTube channel
- Uses auto-generated captions if manual ones aren't available
- Some videos have no captions at all (shorts, music) — these are skipped
- For a 100-video channel, expect 5–15 minutes

### 3. Build the vector database

```bash
python scripts/2_build_vectorstore.py
```

This embeds all transcript chunks using a local free model and stores them in ChromaDB.

**Output:** `vectorstore/` directory + `data/vectorstore_meta.json`

For 100 videos, this takes roughly 2–5 minutes. The vectorstore is reusable — you only rebuild it if you add new videos.

### 4. Set your API key

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...
```

On Windows:
```cmd
set ANTHROPIC_API_KEY=sk-ant-api03-...
```

---

## Running the tutor

### Option A: Web UI (recommended)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

In the sidebar:
1. Enter your Anthropic API key
2. Enter the creator's name (e.g. "Andrej Karpathy")
3. Optionally describe their teaching style
4. Click **Load Tutor**

Then start chatting!

### Option B: Terminal

```bash
python cli.py --creator "Andrej Karpathy"
```

Or ask a single question:
```bash
python cli.py --creator "Andrej Karpathy" --question "Explain backpropagation"
```

---

## Example questions

Once loaded, you can ask anything that creator has covered:

- *"Explain [concept] the way you always do"*
- *"What's your take on [topic]?"*
- *"What tools do you recommend for X?"*
- *"Walk me through how Y works"*
- *"What did you say about Z in your videos?"*

---

## Adding new videos

When the channel releases new content:

```bash
# Re-run extraction (--resume skips already-downloaded videos)
python scripts/1_extract_transcripts.py \
  --channel "https://www.youtube.com/@Handle/videos" \
  --resume

# Rebuild the vector database
python scripts/2_build_vectorstore.py
```

---

## Project structure

```
yt_tutor_agent/
├── app.py                          ← Streamlit web UI
├── cli.py                          ← Terminal interface
├── rag_engine.py                   ← Core RAG logic (retrieval + Claude)
├── requirements.txt
├── scripts/
│   ├── 1_extract_transcripts.py    ← YouTube → transcripts_raw.json
│   └── 2_build_vectorstore.py      ← transcripts → ChromaDB
├── data/
│   ├── transcripts_raw.json        ← raw transcripts (created by step 1)
│   ├── vectorstore_meta.json       ← metadata (created by step 2)
│   └── chunks_preview.json         ← first 20 chunks for inspection
└── vectorstore/                    ← ChromaDB files (created by step 2)
```

---

## Customising the tutor persona

In `app.py` sidebar or via `--style` CLI flag, describe the creator's style. For example:

> *"Builds intuition from first principles before showing code. Uses lots of analogies. Prefers NumPy over PyTorch for teaching. Talks through everything step by step."*

This gets injected into Claude's system prompt to shape how it responds.

---

## Cost estimate

Each question costs roughly $0.002–0.005 USD (Claude Sonnet 4).
A 100-question session costs under $0.50.

---

## Troubleshooting

**"No transcript available" for many videos**
Some channels disable captions. Try adding `--subtitleslangs auto` or check if the channel has manual captions.

**Slow embedding (step 2)**
Normal — the first run downloads the model. Subsequent runs use the cache and are much faster.

**"collection not found" error**
Re-run `scripts/2_build_vectorstore.py`.

**Bad answer quality**
Try increasing `TOP_K` in `rag_engine.py` from 8 to 12. Also make sure the question uses vocabulary the creator actually uses.