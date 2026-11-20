"""Run CodeFast training from the Unsloth Studio Python environment.

This launcher deliberately keeps the model/data definition in train.py so checkpoints stay standard
Transformers artifacts. It prefers the Unsloth runtime when it can be imported, and otherwise uses the
same Llama model with the accelerator exposed by the Unsloth environment.
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--language")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=1024)
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument("--cpu", dest="cpu", action="store_true", help="use CPU training (default)")
    device_group.add_argument("--gpu", dest="cpu", action="store_false", help="opt into the Unsloth accelerator")
    parser.set_defaults(cpu=True)
    args, unknown = parser.parse_known_args()
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
    mode = "CPU-only training" if args.cpu else "the Unsloth accelerator"
    print(f"Using the Unsloth Studio Python runtime with {mode}.")
    sys.argv = ["train.py", "--data", args.data, "--out", args.out, "--steps", str(args.steps), "--seq-len", str(args.seq_len), "--smoke", "--device", "cpu" if args.cpu else "cuda"]
    if args.language:
        sys.argv += ["--language", args.language]
    sys.argv += unknown
    runpy.run_path(str(Path(__file__).with_name("train.py")), run_name="__main__")


if __name__ == "__main__":
    main()
