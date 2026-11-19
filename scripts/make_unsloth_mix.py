"""Build a shuffled Unsloth SFT text corpus from real and synthetic coding data."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def rows(path: Path):
    with path.open(encoding='utf-8', errors='ignore') as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get('text') or row.get('content') or row.get('code')
            if isinstance(text, str) and text.strip():
                yield {'text': text.strip(), 'language': row.get('language', 'code')}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--quality', required=True)
    parser.add_argument('--appstack', required=True)
    parser.add_argument('--multilang', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--app-chunks', type=int, default=20000)
    parser.add_argument('--multilang-rows', type=int, default=10000)
    parser.add_argument('--chunk-chars', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=1337)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    selected = list(rows(Path(args.quality)))
    app_chunks = []
    for row in rows(Path(args.appstack)):
        text = row['text']
        for start in range(0, len(text), args.chunk_chars):
            chunk = text[start:start + args.chunk_chars].strip()
            if len(chunk) >= 120:
                app_chunks.append({'text': chunk, 'language': row['language']})
    rng.shuffle(app_chunks)
    selected.extend(app_chunks[:args.app_chunks])
    web = list(rows(Path(args.multilang)))
    rng.shuffle(web)
    selected.extend(web[:args.multilang_rows])
    rng.shuffle(selected)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps({'output': str(output), 'rows': len(selected), 'app_chunks': min(args.app_chunks, len(app_chunks)), 'quality_rows': len(list(rows(Path(args.quality)))), 'multilang_rows': min(args.multilang_rows, len(web))}, indent=2))


if __name__ == '__main__':
    main()
