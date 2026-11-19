"""Run a lightweight multi-language completion smoke gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM


TASKS = [
    {"name": "python_add", "prompt": "def add_numbers(a, b):\n", "must_contain": ["return"]},
    {"name": "python_async", "prompt": "async def fetch_json(client, url):\n", "must_contain": ["await", "return"]},
    {"name": "javascript_add", "prompt": "function addNumbers(a, b) {\n", "must_contain": ["return", "}"]},
    {"name": "javascript_async", "prompt": "async function loadJson(url) {\n", "must_contain": ["await", "fetch", "return"]},
    {"name": "typescript_user", "prompt": "type User = {\n", "must_contain": ["id", "string"]},
    {"name": "typescript_handler", "prompt": "export async function handler(req: Request): Promise<Response> {\n", "must_contain": ["return", "Response"]},
    {"name": "sql_users", "prompt": "SELECT u.id, u.email\nFROM users u\n", "must_contain": ["WHERE"]},
    {"name": "html_form", "prompt": "<form method=\"post\" action=\"/login\">\n", "must_contain": ["input", "</form>"]},
    {"name": "css_card", "prompt": ".card {\n", "must_contain": ["display", "padding"]},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    results = []
    for task in TASKS:
        ids = tokenizer.encode(task["prompt"]).ids
        inputs = torch.tensor([ids], device=device)
        with torch.no_grad():
            output = model.generate(
                inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=model.config.eos_token_id,
                pad_token_id=model.config.pad_token_id,
            )
        text = tokenizer.decode(output[0].tolist())
        missing = [item for item in task["must_contain"] if item.lower() not in text.lower()]
        results.append({"name": task["name"], "passed": not missing, "missing": missing, "generated": text})
    report = {"examples": len(results), "passed": sum(item["passed"] for item in results), "results": results}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"examples": report["examples"], "passed": report["passed"]}, indent=2))


if __name__ == "__main__":
    main()
