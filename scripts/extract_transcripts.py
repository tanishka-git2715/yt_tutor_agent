"""
STEP 1 — Extract transcripts from a YouTube channel
----------------------------------------------------
Usage:
    python scripts/1_extract_transcripts.py --channel "https://www.youtube.com/@ChannelHandle/videos"

Output:
    data/transcripts_raw.json   — list of {video_id, title, url, transcript}
"""

import argparse
import json
import os
import re
import sys
import time

def check_dependencies():
    try:
        import yt_dlp
    except ImportError:
        print("Installing yt-dlp...")
        os.system(f"{sys.executable} -m pip install yt-dlp -q")

check_dependencies()

import yt_dlp  # noqa: E402

OUTPUT_FILE = "data/transcripts_raw.json"
os.makedirs("data", exist_ok=True)
os.makedirs("data/tmp", exist_ok=True)


def clean_text(text: str) -> str:
    """Remove filler, timestamps, and normalise whitespace."""
    text = re.sub(r"\[.*?\]", "", text)          # remove [Music], [Applause] etc
    text = re.sub(r"\s+", " ", text)             # collapse whitespace
    return text.strip()


def get_all_video_ids(channel_url: str) -> list[dict]:
    """Return list of {id, title} for every video on the channel."""
    opts = {
        "quiet": True,
        "extract_flat": "in_playlist",
        "skip_download": True,
        "ignoreerrors": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)

    if not info:
        raise RuntimeError(f"Could not fetch channel info from: {channel_url}")

    entries = info.get("entries") or []
    videos = []
    for e in entries:
        if e and e.get("id"):
            videos.append({"id": e["id"], "title": e.get("title", "Untitled")})
    return videos


def fetch_transcript_for_video(video_id: str) -> str | None:
    """
    Fetch transcript text for a single video using yt-dlp.
    Prefers manual English captions, falls back to auto-generated.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    opts = {
        "quiet": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-US", "en-GB"],
        "subtitlesformat": "json3",
        "outtmpl": f"data/tmp/yt_sub_{video_id}.%(ext)s",
        "ignoreerrors": True,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    # Find the downloaded subtitle file
    for lang in ["en", "en-US", "en-GB"]:
        for kind in ["", ".auto"]:
            filepath = f"data/tmp/yt_sub_{video_id}{kind}.{lang}.json3"
            # yt-dlp names files differently across versions — try both patterns
            alt_path = f"data/tmp/yt_sub_{video_id}.{lang}.json3"
            for path in [filepath, alt_path]:
                if os.path.exists(path):
                    text = parse_json3(path)
                    os.remove(path)
                    return text

    return None


def parse_json3(filepath: str) -> str:
    """Convert yt-dlp json3 subtitle format to plain text."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    parts = []
    for event in data.get("events", []):
        for seg in event.get("segs", []):
            t = seg.get("utf8", "").strip()
            if t and t != "\n":
                parts.append(t)
    return clean_text(" ".join(parts))


def main():
    parser = argparse.ArgumentParser(description="Extract YouTube channel transcripts")
    parser.add_argument("--channel", required=True, help="Channel URL, e.g. https://www.youtube.com/@Handle/videos")
    parser.add_argument("--limit", type=int, default=0, help="Max videos to process (0 = all)")
    parser.add_argument("--resume", action="store_true", help="Skip videos already in output file")
    args = parser.parse_args()

    # Load existing data if resuming
    existing = {}
    if args.resume and os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            for item in json.load(f):
                existing[item["video_id"]] = item
        print(f"Resuming — {len(existing)} videos already extracted")

    print(f"\nFetching video list from: {args.channel}")
    videos = get_all_video_ids(args.channel)
    if not videos:
        print("No videos found. Check the channel URL.")
        return

    print(f"Found {len(videos)} videos")
    if args.limit:
        videos = videos[: args.limit]
        print(f"Limiting to {args.limit} videos")

    results = list(existing.values())
    success = len(existing)
    failed = 0

    for i, v in enumerate(videos):
        vid = v["id"]
        if vid in existing:
            continue

        print(f"[{i+1}/{len(videos)}] {v['title'][:70]}")
        transcript = fetch_transcript_for_video(vid)

        if transcript and len(transcript) > 100:
            results.append({
                "video_id": vid,
                "title": v["title"],
                "url": f"https://www.youtube.com/watch?v={vid}",
                "transcript": transcript,
                "char_count": len(transcript),
            })
            success += 1
            print(f"         OK — {len(transcript):,} chars")
        else:
            failed += 1
            print(f"         SKIP — no transcript available")

        # Save incrementally every 10 videos
        if (i + 1) % 10 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        time.sleep(0.5)  # be polite to YouTube

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDONE: {success} transcripts saved -> {OUTPUT_FILE}")
    print(f"  {failed} videos had no available transcript")


if __name__ == "__main__":
    main()
