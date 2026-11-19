"""Distill CodeGPT-small-py into the 99.76M rotary/64k student."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tokenizers import Tokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, LlamaForCausalLM

from train import PackedTokens, clean_documents, count_parameters, read_documents, select_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher", default="microsoft/CodeGPT-small-py")
    parser.add_argument("--config", default="configs/code_100m_codegpt_vocab.yaml")
    parser.add_argument("--language", default="Python")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--ce-weight", type=float, default=0.25)
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(Path(args.student) / "tokenizer.json"))
    student = LlamaForCausalLM.from_pretrained(args.student)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student.to(device); teacher.to(device)
    optimizer = AdamW(student.parameters(), lr=args.learning_rate, weight_decay=0.05, betas=(0.9, 0.95))
    raw = lambda: read_documents(args.data, args.language, cfg["data"]["seed"])
    cleaned = lambda: clean_documents(raw(), cfg["data"]["min_chars"], cfg["data"]["max_chars"])
    dataset = PackedTokens(
        lambda: select_split(cleaned(), validation=False), tokenizer, args.seq_len, cfg["data"]["seed"],
        cfg["model"]["bos_token_id"], cfg["model"]["eos_token_id"],
    )
    loader = iter(DataLoader(dataset, batch_size=1, num_workers=0))
    teacher_vocab = teacher.config.vocab_size
    total_tokens = 0; started = time.time(); student.train()
    for step in range(1, args.steps + 1):
        batch = next(loader)
        input_ids = batch["input_ids"].to(device); labels = batch["labels"].to(device)
        with torch.no_grad():
            teacher_logits = teacher(input_ids=input_ids).logits
        student_output = student(input_ids=input_ids, labels=labels)
        student_logits = student_output.logits[..., :teacher_vocab]
        temperature = args.temperature
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2) / input_ids.shape[1]
        loss = args.ce_weight * student_output.loss + (1.0 - args.ce_weight) * distill_loss
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step(); total_tokens += input_ids.numel()
        if step % 10 == 0:
            print(f"step {step}/{args.steps} loss={loss.item():.4f} ce={student_output.loss.item():.4f} kd={distill_loss.item():.4f} tokens/s={total_tokens/max(1e-6,time.time()-started):.0f}", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            checkpoint = output / f"step-{step}"; checkpoint.mkdir(exist_ok=True)
            student.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save(str(checkpoint / "tokenizer.json"))
            (checkpoint / "training_state.json").write_text(json.dumps({"step": step, "tokens": total_tokens, "loss": loss.item()}, indent=2), encoding="utf-8")
    final = output / "final"; student.save_pretrained(final, safe_serialization=True); tokenizer.save(str(final / "tokenizer.json"))
    report = {"teacher": args.teacher, "parameters": count_parameters(student), "context_length": student.config.max_position_embeddings, "steps": args.steps, "tokens_seen": total_tokens, "sequence_length": args.seq_len}
    (output / "distillation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
