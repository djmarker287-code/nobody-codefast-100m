"""Create a small deterministic code corpus for smoke-testing the full pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


TEMPLATES = [
    """def {name}(items):\n    \"\"\"Return the transformed items in stable order.\"\"\"\n    result = []\n    for item in items:\n        if item is not None:\n            result.append({expr})\n    return result\n""",
    """class {name}:\n    def __init__(self, values=None):\n        self.values = list(values or [])\n\n    def add(self, value):\n        self.values.append(value)\n        return self\n\n    def total(self):\n        return sum(self.values)\n""",
    """def {name}(text, limit=80):\n    text = text.strip()\n    if len(text) <= limit:\n        return text\n    return text[: limit - 3].rstrip() + \"...\"\n""",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="work/demo.jsonl")
    parser.add_argument("--documents", type=int, default=2000)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index in range(args.documents):
            template = TEMPLATES[index % len(TEMPLATES)]
            text = template.format(
                name=f"transform_{index}",
                expr="item * 2" if index % 2 else "item.upper()",
            )
            handle.write(json.dumps({"text": text, "language": "Python"}) + "\n")
    print(f"wrote {args.documents} documents to {output}")


if __name__ == "__main__":
    main()
