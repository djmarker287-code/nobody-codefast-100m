"""Train CodeFast-100M from local JSONL or a streaming Hugging Face dataset."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator

import torch
import yaml
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, GPTNeoXConfig, GPTNeoXForCausalLM, LlamaConfig, LlamaForCausalLM


SECRET_RE = re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"]+['\"]")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_documents(source: str | list[str], language: str | None, seed: int, dataset_config: str | None = None) -> Iterator[str]:
    if isinstance(source, list):
        for item in source:
            yield from read_documents(item, language, seed, dataset_config)
        return
    path = Path(source)
    if path.exists():
        files = [path] if path.is_file() else sorted(path.rglob("*.jsonl"))
        for file in files:
            with file.open(encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if language and str(row.get("language", "")).lower() != language.lower():
                        continue
                    text = row.get("text") or row.get("content") or row.get("code")
                    if isinstance(text, str):
                        yield text
        return
    from datasets import load_dataset

    kwargs = {"split": "train", "streaming": True}
    if dataset_config:
        kwargs["name"] = dataset_config
    dataset = load_dataset(source, **kwargs)
    rows = list(dataset.take(1))
    if rows and "content" not in rows[0] and "text" not in rows[0] and "code" not in rows[0]:
        raise ValueError(f"Dataset {source!r} has no text/content/code field")
    for row in dataset:
        if language and str(row.get("language", "")).lower() != language.lower():
            continue
        text = row.get("text") or row.get("content") or row.get("code")
        if isinstance(text, str):
            yield text


def clean_documents(documents: Iterable[str], min_chars: int, max_chars: int) -> Iterator[str]:
    for text in documents:
        text = text.replace("\x00", "").strip()
        if len(text) < min_chars or len(text) > max_chars or "\\x00" in text:
            continue
        if SECRET_RE.search(text):
            continue
        printable = sum(ch.isprintable() or ch in "\n\t\r" for ch in text) / max(1, len(text))
        if printable < 0.92:
            continue
        yield text


def select_split(documents: Iterable[str], validation: bool, period: int = 100) -> Iterator[str]:
    """Deterministically reserve one document in every period for validation."""
    for index, document in enumerate(documents):
        is_validation = index % period == 0
        if is_validation == validation:
            yield document


def train_tokenizer(documents: Iterable[str], output: Path, vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(documents, trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))
    return tokenizer


class PackedTokens(IterableDataset):
    def __init__(self, documents: Iterable[str] | Callable[[], Iterable[str]], tokenizer: Tokenizer, seq_len: int, seed: int, bos_id: int = 1, eos_id: int = 2):
        self.documents, self.tokenizer, self.seq_len, self.seed = documents, tokenizer, seq_len, seed
        self.bos_id, self.eos_id = bos_id, eos_id

    def __iter__(self):
        while True:
            buffer: list[int] = []
            produced = False
            documents = self.documents() if callable(self.documents) else self.documents
            for text in documents:
                encoded = self.tokenizer.encode(text).ids
                buffer.extend([self.bos_id, *encoded, self.eos_id])
                while len(buffer) >= self.seq_len + 1:
                    chunk, buffer = buffer[: self.seq_len + 1], buffer[self.seq_len + 1 :]
                    produced = True
                    yield {"input_ids": torch.tensor(chunk[:-1], dtype=torch.long), "labels": torch.tensor(chunk[1:], dtype=torch.long)}
            if not callable(self.documents) or not produced:
                break


def build_model(cfg: dict):
    m = cfg["model"]
    if m.get("architecture") == "gpt_neox":
        return GPTNeoXForCausalLM(GPTNeoXConfig(
            vocab_size=m["vocab_size"], max_position_embeddings=m["context_length"],
            hidden_size=m["hidden_size"], num_hidden_layers=m["num_layers"],
            num_attention_heads=m["num_heads"], intermediate_size=m["intermediate_size"],
            hidden_dropout=m["dropout"], attention_dropout=m["dropout"],
            rotary_pct=m.get("rotary_pct", 1.0), rotary_emb_base=m["rope_theta"],
            hidden_act=m.get("hidden_act", "gelu"),
            use_parallel_residual=m.get("use_parallel_residual", False),
            bos_token_id=m.get("bos_token_id", 0), eos_token_id=m.get("eos_token_id", 2),
            pad_token_id=m.get("pad_token_id", 1), tie_word_embeddings=m.get("tie_word_embeddings", True),
        ))
    config = LlamaConfig(
        vocab_size=m["vocab_size"], max_position_embeddings=m["context_length"],
        hidden_size=m["hidden_size"], num_hidden_layers=m["num_layers"],
        num_attention_heads=m["num_heads"], num_key_value_heads=m["num_heads"],
        intermediate_size=m["intermediate_size"], attention_dropout=m["dropout"],
        rope_theta=m["rope_theta"], bos_token_id=m.get("bos_token_id", 1), eos_token_id=m.get("eos_token_id", 2), pad_token_id=m.get("pad_token_id", 0),
        tie_word_embeddings=m.get("tie_word_embeddings", False),
    )
    return LlamaForCausalLM(config)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, limit: int, autocast_dtype):
    model.eval(); losses = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= limit: break
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                loss = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device)).loss
            losses.append(loss.detach().float().item())
    model.train()
    mean = sum(losses) / max(1, len(losses))
    return mean, math.exp(min(mean, 20))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--out", default="outputs/codefast-100m")
    parser.add_argument("--config", default="configs/code_100m.yaml")
    parser.add_argument("--language")
    parser.add_argument("--dataset-config", help="Hugging Face dataset configuration/name, when needed")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--seq-len", type=int, help="training sequence length; defaults to 1024 for smoke, otherwise 65536")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.smoke:
        cfg["training"].update(max_steps=100, eval_every=50, save_every=50, log_every=10, batch_size=1, grad_accumulation=1)
    if args.steps: cfg["training"]["max_steps"] = args.steps
    if args.batch_size: cfg["training"]["batch_size"] = args.batch_size
    if args.grad_accum: cfg["training"]["grad_accumulation"] = args.grad_accum
    seq_len = args.seq_len or (1024 if args.smoke else cfg["model"]["context_length"])
    if seq_len > cfg["model"]["context_length"]:
        raise ValueError(f"--seq-len {seq_len} exceeds model context {cfg['model']['context_length']}")
    seed_everything(cfg["training"]["seed"])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    raw = lambda: read_documents(args.data, args.language, cfg["data"]["seed"], args.dataset_config)
    cleaned = lambda: clean_documents(raw(), cfg["data"]["min_chars"], cfg["data"]["max_chars"])
    tokenizer_path = out / "tokenizer.json"
    if tokenizer_path.exists(): tokenizer = Tokenizer.from_file(str(tokenizer_path))
    elif args.resume and (Path(args.resume) / "tokenizer.json").exists():
        tokenizer = Tokenizer.from_file(str(Path(args.resume) / "tokenizer.json"))
        tokenizer.save(str(tokenizer_path))
    else:
        print("training tokenizer...")
        tokenizer = train_tokenizer((doc for _, doc in zip(range(20000), cleaned())), tokenizer_path, cfg["model"]["vocab_size"])
    model = build_model(cfg)
    params = count_parameters(model)
    print(f"trainable parameters: {params:,}")
    if not 95_000_000 <= params <= 105_000_000:
        raise RuntimeError(f"model is not approximately 100M parameters: {params}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = cfg["training"]["precision"]
    if precision == "auto": precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else ("fp16" if device.type == "cuda" else "fp32")
    autocast_dtype = torch.bfloat16 if precision == "bf16" else (torch.float16 if precision == "fp16" else None)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"], betas=(0.9, 0.95))
    def lr_factor(step: int) -> float:
        warmup = cfg["training"]["warmup_steps"]
        total = cfg["training"]["max_steps"]
        if step < warmup:
            return max(1e-8, step / max(1, warmup))
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        floor = cfg["training"]["min_learning_rate"] / cfg["training"]["learning_rate"]
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16" and device.type == "cuda")
    train_ds = PackedTokens(lambda: select_split(cleaned(), validation=False), tokenizer, seq_len, cfg["data"]["seed"], cfg["model"].get("bos_token_id", 1), cfg["model"].get("eos_token_id", 2))
    val_ds = PackedTokens(lambda: select_split(cleaned(), validation=True), tokenizer, seq_len, cfg["data"]["seed"], cfg["model"].get("bos_token_id", 1), cfg["model"].get("eos_token_id", 2))
    loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], num_workers=0)
    iterator = iter(loader)
    total_tokens = 0; started = time.time(); optimizer.zero_grad(set_to_none=True)
    start_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        resumed = AutoModelForCausalLM.from_pretrained(resume_path)
        model.load_state_dict(resumed.state_dict())
        del resumed
        state_path = resume_path / "training_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            start_step, total_tokens = int(state.get("step", 0)), int(state.get("tokens", 0))
        print(f"resuming from step {start_step}")
    max_steps = cfg["training"]["max_steps"]
    for step in range(start_step + 1, max_steps + 1):
        batch = next(iterator)
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            loss = model(**batch).loss / cfg["training"]["grad_accumulation"]
        if scaler.is_enabled(): scaler.scale(loss).backward()
        else: loss.backward()
        if step % cfg["training"]["grad_accumulation"] == 0:
            if scaler.is_enabled(): scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["max_grad_norm"])
            if scaler.is_enabled(): scaler.step(optimizer); scaler.update()
            else: optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total_tokens += batch["input_ids"].numel()
        if step % cfg["training"]["log_every"] == 0:
            elapsed = max(1e-6, time.time() - started)
            print(f"step {step}/{max_steps} loss={loss.item() * cfg['training']['grad_accumulation']:.4f} tokens/s={total_tokens/elapsed:.0f} device={device} precision={precision}")
        if step % cfg["training"]["eval_every"] == 0:
            val_loss, val_ppl = evaluate(model, val_loader, device, limit=4, autocast_dtype=autocast_dtype)
            print(f"eval step {step} val_loss={val_loss:.4f} val_perplexity={val_ppl:.2f}")
        if args.tokens and total_tokens >= args.tokens:
            print(f"token budget reached: {total_tokens}")
            max_steps = step
            break
        if step % cfg["training"]["save_every"] == 0 or step == max_steps:
            ckpt = out / f"step-{step}"
            ckpt.mkdir(exist_ok=True)
            model.save_pretrained(ckpt, safe_serialization=True)
            tokenizer.save(str(ckpt / "tokenizer.json"))
            (ckpt / "training_state.json").write_text(json.dumps({"step": step, "tokens": total_tokens, "parameters": params}, indent=2), encoding="utf-8")
    final = out / "final"; model.save_pretrained(final, safe_serialization=True); tokenizer.save(str(final / "tokenizer.json"))
    card = {"name": "CodeFast-100M", "parameters": params, "tokens_seen": total_tokens, "device": str(device), "precision": precision, "config": cfg}
    (out / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
