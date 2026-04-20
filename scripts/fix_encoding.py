import os

def fix_json_encoding(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    # Read the raw bytes
    with open(filepath, "rb") as f:
        raw_data = f.read()

    # Try decoding as UTF-8
    try:
        raw_data.decode("utf-8")
        print(f"File {filepath} is already valid UTF-8.")
        return
    except UnicodeDecodeError:
        print(f"File {filepath} has invalid UTF-8 bytes. Attempting conversion from cp1252...")

    # Decode as cp1252 (Windows default) and re-encode as UTF-8
    try:
        decoded_text = raw_data.decode("cp1252")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(decoded_text)
        print(f"Successfully converted {filepath} to UTF-8.")
    except Exception as e:
        print(f"Failed to convert {filepath}: {e}")

if __name__ == "__main__":
    fix_json_encoding("data/transcripts_raw.json")
