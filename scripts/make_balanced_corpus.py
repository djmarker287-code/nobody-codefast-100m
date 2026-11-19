"""Build a balanced, chunked curriculum corpus from coding sources."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def read_rows(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text") or row.get("content") or row.get("code")
            if isinstance(text, str) and text.strip():
                rows.append({"text": text.strip(), "language": row.get("language", "Unknown")})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", required=True)
    parser.add_argument("--multilang", required=True)
    parser.add_argument("--appstack", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-chars", type=int, default=24_000_000)
    parser.add_argument("--chunk-chars", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    quality = read_rows(Path(args.quality))
    multilang = read_rows(Path(args.multilang))
    app_rows = read_rows(Path(args.appstack))
    app_chunks = []
    for row in app_rows:
        text = row["text"]
        for start in range(0, len(text), args.chunk_chars):
            chunk = text[start : start + args.chunk_chars].strip()
            if len(chunk) >= 80:
                app_chunks.append({"text": chunk, "language": row["language"]})
    rng.shuffle(app_chunks)

    app_target = args.target_chars // 2
    small_target = args.target_chars - app_target
    selected = []
    total = 0
    for row in app_chunks:
        if total >= app_target:
            break
        selected.append(row); total += len(row["text"])

    small = quality + multilang
    rng.shuffle(small)
    total = 0; index = 0
    while total < small_target:
        row = small[index % len(small)]
        selected.append(row); total += len(row["text"]); index += 1
    rng.shuffle(selected)

    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"documents": len(selected), "characters": sum(len(row["text"]) for row in selected), "app_documents": len(app_chunks), "quality_documents": len(quality), "multilang_documents": len(multilang)}, indent=2))


if __name__ == "__main__":
    main()
