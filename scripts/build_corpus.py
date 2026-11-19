"""Pack source files from cloned repositories into filtered JSONL training data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXTENSIONS = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript", ".ts": "TypeScript",
    ".tsx": "TypeScript", ".java": "Java", ".go": "Go", ".rs": "Rust", ".c": "C",
    ".h": "C", ".cpp": "C++", ".cc": "C++", ".cs": "C#", ".rb": "Ruby", ".php": "PHP",
    ".html": "HTML", ".css": "CSS", ".sql": "SQL", ".json": "JSON", ".yaml": "YAML",
    ".yml": "YAML", ".toml": "TOML", ".md": "Markdown",
}
SKIP_PARTS = {".git", "node_modules", "vendor", "dist", "build", "target", "__pycache__", ".venv"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-bytes", type=int, default=1_000_000)
    args = parser.parse_args()
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set(); count = 0; bytes_written = 0
    with output.open("w", encoding="utf-8") as handle:
        for root_name in args.roots:
            root = Path(root_name)
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                    continue
                if any(part in SKIP_PARTS for part in path.parts):
                    continue
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore").strip()
                except OSError:
                    continue
                if len(content) < 120 or len(content.encode("utf-8")) > args.max_bytes:
                    continue
                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                row = {"text": content, "language": EXTENSIONS[path.suffix.lower()], "source": str(path)}
                encoded = json.dumps(row, ensure_ascii=False) + "\n"
                handle.write(encoded); count += 1; bytes_written += len(encoded.encode("utf-8"))
    print(f"wrote {count} documents ({bytes_written:,} bytes) to {output}")


if __name__ == "__main__":
    main()
