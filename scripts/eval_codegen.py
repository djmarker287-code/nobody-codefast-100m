"""Evaluate basic Python syntax validity on prompts from a JSONL code corpus."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import LlamaForCausalLM


def prompts_from_jsonl(path: Path, limit: int):
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if count >= limit:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text", "")
            lines = text.splitlines()
            for index, value in enumerate(lines):
                if value.startswith("def ") or value.startswith("class "):
                    prompt = "\n".join(lines[: index + 1])
                    if len(prompt) >= 20:
                        yield prompt
                        count += 1
                        break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    model = LlamaForCausalLM.from_pretrained(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    results = []
    for prompt in prompts_from_jsonl(Path(args.data), args.limit):
        ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
        with torch.no_grad():
            generated = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False, eos_token_id=2, pad_token_id=0)
        text = tokenizer.decode(generated[0].tolist())
        try:
            ast.parse(text)
            valid = True
            error = None
        except SyntaxError as exc:
            valid = False
            error = f"{exc.msg} at line {exc.lineno}"
        results.append({"prompt": prompt, "generated": text, "syntax_valid": valid, "error": error})
    report = {"checkpoint": str(checkpoint), "examples": len(results), "syntax_valid": sum(item["syntax_valid"] for item in results), "results": results}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("checkpoint", "examples", "syntax_valid")}, indent=2))


if __name__ == "__main__":
    main()
