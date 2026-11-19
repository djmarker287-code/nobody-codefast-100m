"""Generate Python code and keep the longest executable prefix."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList


def longest_valid_prefix(source: str) -> str:
    lines = source.splitlines()
    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end]).rstrip() + "\n"
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue
    return ""


class ValidPythonStopping(StoppingCriteria):
    """Stop immediately after a generated prefix becomes valid Python."""

    def __init__(self, tokenizer: Tokenizer, prompt_tokens: int):
        self.tokenizer = tokenizer
        self.prompt_tokens = prompt_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[1] <= self.prompt_tokens:
            return False
        text = self.tokenizer.decode(input_ids[0].tolist())
        if "def " not in text and "class " not in text:
            return False
        try:
            ast.parse(text)
            last_line = text.rstrip().splitlines()[-1].strip()
            return last_line not in {"return", "yield"}
        except SyntaxError:
            return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--no-early-stop", action="store_true")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    prompt_ids = tokenizer.encode(args.prompt).ids
    ids = torch.tensor([prompt_ids], device=device)
    stopping = None if args.no_early_stop else StoppingCriteriaList([ValidPythonStopping(tokenizer, len(prompt_ids))])
    with torch.no_grad():
        output = model.generate(
            ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=model.config.eos_token_id,
            pad_token_id=model.config.pad_token_id,
            stopping_criteria=stopping,
        )
    raw = tokenizer.decode(output[0].tolist())
    checked = longest_valid_prefix(raw)
    print(json.dumps({
        "raw": raw,
        "checked": checked,
        "valid_python": bool(checked),
        "trimmed": bool(checked) and checked != raw,
    }, indent=2))


if __name__ == "__main__":
    main()
