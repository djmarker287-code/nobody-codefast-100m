"""Prefix a normal coding corpus with repeated teacher completions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()
    teacher = read(Path(args.teacher)); base = read(Path(args.base))
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as handle:
        for _ in range(args.repeats):
            for row in teacher:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        for row in base:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps({'output': str(output), 'teacher_rows': len(teacher), 'base_rows': len(base), 'teacher_repeats': args.repeats}, indent=2))


if __name__ == '__main__':
    main()
