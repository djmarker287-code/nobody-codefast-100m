"""Run CodeFast with an automatic Python/web checkpoint router."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import LlamaForCausalLM, StoppingCriteria, StoppingCriteriaList

from generate_checked import longest_valid_prefix


class ValidPythonStopping(StoppingCriteria):
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
            return text.rstrip().splitlines()[-1].strip() not in {"return", "yield"}
        except SyntaxError:
            return False


def is_web_prompt(prompt: str) -> bool:
    text = prompt.lower()
    markers = ("function ", "async function", "typescript", "interface ", "type ", "export ", "import react", "select ", "<form", ".card", "display:")
    return any(marker in text for marker in markers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="default full-stack checkpoint")
    parser.add_argument("--web-checkpoint", required=True, help="specialized web checkpoint")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    chosen = Path(args.web_checkpoint if is_web_prompt(args.prompt) else args.checkpoint)
    tokenizer = Tokenizer.from_file(str(chosen / "tokenizer.json"))
    model = LlamaForCausalLM.from_pretrained(chosen)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    prompt_ids = tokenizer.encode(args.prompt).ids
    inputs = torch.tensor([prompt_ids], device=device)
    stopping = None
    if chosen == Path(args.checkpoint) and ("def " in args.prompt or "class " in args.prompt):
        stopping = StoppingCriteriaList([ValidPythonStopping(tokenizer, len(prompt_ids))])
    with torch.no_grad():
        output = model.generate(inputs, max_new_tokens=args.max_new_tokens, do_sample=False, eos_token_id=model.config.eos_token_id, pad_token_id=model.config.pad_token_id, stopping_criteria=stopping)
    raw = tokenizer.decode(output[0].tolist())
    checked = longest_valid_prefix(raw) if ("def " in args.prompt or "class " in args.prompt) else ""
    print(json.dumps({
        "checkpoint": str(chosen),
        "raw": raw,
        "checked": checked,
        "valid_python": bool(checked) if ("def " in args.prompt or "class " in args.prompt) else None,
    }, indent=2))


if __name__ == "__main__":
    main()
