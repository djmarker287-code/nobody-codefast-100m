"""Generate code from a trained CodeFast checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import LlamaForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    model = LlamaForCausalLM.from_pretrained(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    ids = tokenizer.encode(args.prompt).ids
    input_ids = torch.tensor([ids], device=device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=args.temperature, top_p=args.top_p, eos_token_id=2, pad_token_id=0)
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
