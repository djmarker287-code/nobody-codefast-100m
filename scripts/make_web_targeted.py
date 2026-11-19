"""Create repeated, exact web completion patterns for specialist calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


PATTERNS = [
    ("function addNumbers(a, b) {\n  return a + b;\n}\n", "JavaScript"),
    ("async function loadJson(url) {\n  const response = await fetch(url);\n  return response.json();\n}\n", "JavaScript"),
    ("type User = {\n  id: number;\n  name: string;\n};\n", "TypeScript"),
    ("export async function handler(req: Request): Promise<Response> {\n  return new Response('ok');\n}\n", "TypeScript"),
    ("export function Button({ label }: { label: string }) {\n  return <button>{label}</button>;\n}\n", "TypeScript"),
    ("SELECT id, email FROM users WHERE active = true ORDER BY id;\n", "SQL"),
    ("<form method=\"post\" action=\"/login\">\n  <input name=\"email\" type=\"email\" />\n  <button type=\"submit\">Log in</button>\n</form>\n", "HTML"),
    (".card {\n  display: flex;\n  padding: 1rem;\n  border-radius: 0.5rem;\n}\n", "CSS"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=3000)
    args = parser.parse_args()
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for _ in range(args.repeats):
            for text, language in PATTERNS:
                handle.write(json.dumps({"text": text, "language": language}) + "\n")
    print(f"wrote {len(PATTERNS) * args.repeats} targeted examples")


if __name__ == "__main__":
    main()
