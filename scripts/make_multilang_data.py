"""Create deterministic multi-language web/app coding data and a held-out smoke set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def examples():
    for i in range(3000):
        yield f"function addNumbers_{i}(a, b) {{\n  return a + b;\n}}\n", "JavaScript"
        yield f"async function loadJson_{i}(url) {{\n  const response = await fetch(url);\n  return response.json();\n}}\n", "JavaScript"
        yield f"type User_{i} = {{\n  id: number;\n  name: string;\n}};\n", "TypeScript"
        yield f"export async function handler_{i}(req: Request): Promise<Response> {{\n  return new Response('ok');\n}}\n", "TypeScript"
        yield f"export function Button_{i}({{ label }}: {{ label: string }}) {{\n  return <button>{{label}}</button>;\n}}\n", "TypeScript"
        yield f"SELECT id, email FROM users_{i} WHERE active = true ORDER BY id;\n", "SQL"
        yield f"<form method=\"post\" action=\"/login/{i}\">\n  <input name=\"email\" type=\"email\" />\n  <button type=\"submit\">Log in</button>\n</form>\n", "HTML"
        yield f".card_{i} {{\n  display: flex;\n  padding: 1rem;\n  border-radius: 0.5rem;\n}}\n", "CSS"
        yield f"services:\n  api_{i}:\n    image: example/api\n    ports:\n      - \"8000:8000\"\n", "YAML"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--eval-count", type=int, default=72)
    args = parser.parse_args()
    rows = list(examples())
    held_out = min(args.eval_count, len(rows))
    train_path, eval_path = Path(args.train), Path(args.eval)
    train_path.parent.mkdir(parents=True, exist_ok=True); eval_path.parent.mkdir(parents=True, exist_ok=True)
    with train_path.open("w", encoding="utf-8") as handle:
        for text, language in rows[:-held_out]:
            handle.write(json.dumps({"text": text, "language": language}) + "\n")
    with eval_path.open("w", encoding="utf-8") as handle:
        for index, (text, language) in enumerate(rows[-held_out:]):
            handle.write(json.dumps({"code": text, "name": f"{language}_{index}", "language": language}) + "\n")
    print(f"wrote {len(rows) - held_out} training examples and {held_out} held-out examples")


if __name__ == "__main__":
    main()
