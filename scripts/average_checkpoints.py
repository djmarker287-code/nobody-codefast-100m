"""Interpolate two compatible CodeFast checkpoints and copy their shared metadata."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--specialist", required=True)
    parser.add_argument("--alpha", type=float, required=True, help="specialist weight from 0 to 1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")

    base = Path(args.base); specialist = Path(args.specialist); output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    base_weights = load_file(str(base / "model.safetensors"), device="cpu")
    specialist_weights = load_file(str(specialist / "model.safetensors"), device="cpu")
    if set(base_weights) != set(specialist_weights):
        raise ValueError("checkpoints do not have matching parameter keys")
    blended = {}
    for key, tensor in base_weights.items():
        if tensor.is_floating_point():
            blended[key] = ((1.0 - args.alpha) * tensor + args.alpha * specialist_weights[key]).to(tensor.dtype)
        else:
            blended[key] = tensor
    save_file(blended, str(output / "model.safetensors"), metadata={"format": "pt", "alpha_specialist": str(args.alpha)})
    for name in ("config.json", "tokenizer.json"):
        shutil.copy2(base / name, output / name)
    card = {"base": str(base), "specialist": str(specialist), "alpha_specialist": args.alpha, "parameters": sum(t.numel() for t in blended.values())}
    (output / "blend_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
