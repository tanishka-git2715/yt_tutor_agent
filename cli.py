"""
cli.py — Command-line version of the YT Tutor
----------------------------------------------
Usage:
    python cli.py
    python cli.py --question "Explain transformers"
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="YouTube Channel AI Tutor (CLI)")
    parser.add_argument("--question", "-q", help="Single question (non-interactive mode)")
    parser.add_argument("--creator", default="the creator", help="Creator name for persona")
    parser.add_argument("--style", default="", help="Creator teaching style description")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable first.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    print("Loading knowledge base...")
    try:
        from rag_engine import YTTutorEngine
        engine = YTTutorEngine(creator_name=args.creator, creator_style=args.style)
        meta = engine.meta
        print(f"Loaded: {meta['total_videos']} videos, {meta['total_chunks']:,} chunks\n")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    def ask_and_print(question: str):
        print(f"\n{'─'*60}")
        print(f"You: {question}")
        print(f"{'─'*60}")
        print(f"{args.creator}: ", end="", flush=True)

        sources, stream_ctx = engine.ask_stream(question)
        with stream_ctx as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
        print()

        print(f"\n  Sources used:")
        seen = set()
        for s in sources:
            if s.title not in seen:
                seen.add(s.title)
                print(f"  • {s.title} ({s.url}) — {s.score:.0%} relevance")

    # Single question mode
    if args.question:
        ask_and_print(args.question)
        return

    # Interactive mode
    print(f"🎓 {args.creator} AI Tutor — type 'quit' to exit\n")
    history = []
    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break

        response = engine.ask(question, chat_history=history)
        print(f"\n{args.creator}: {response.answer}\n")

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": response.answer})

        # Keep last 6 turns
        history = history[-12:]


if __name__ == "__main__":
    main()
