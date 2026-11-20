"""Open a simple interactive CodeFast chat in PowerShell."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import LlamaForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a CodeFast checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but this Python environment has no usable CUDA GPU. Use --device cpu.")

    checkpoint = Path(args.checkpoint)
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    model = LlamaForCausalLM.from_pretrained(checkpoint)
    device = torch.device(args.device)
    model.to(device).eval()
    print(f"Loaded {checkpoint} on {device}. Type 'exit' or 'quit' to stop.")

    while True:
        try:
            prompt = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt:
            continue
        ids = tokenizer.encode(prompt).ids
        input_ids = torch.tensor([ids], device=device)
        generation_kwargs = {"max_new_tokens": args.max_new_tokens, "eos_token_id": 2, "pad_token_id": 0}
        if args.temperature > 0:
            generation_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
        else:
            generation_kwargs["do_sample"] = False
        with torch.no_grad():
            output = model.generate(input_ids, **generation_kwargs)
        answer = tokenizer.decode(output[0, input_ids.shape[1]:].tolist()).strip()
        print(f"Model> {answer}\n")


if __name__ == "__main__":
    main()

