"""Initialize the 99M/64k Llama model from compatible CodeGPT-small-py weights.

The source model has 12x768 GPT-2 blocks and a 50,001-token vocabulary. The target keeps the
requested 99M parameter budget and 65,536 rotary context. Compatible attention, norms, truncated
MLP projections, and vocabulary embeddings are transferred; the target remains a standard
Transformers checkpoint and can be continued with scripts/train.py --resume.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from train import build_model, count_parameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/code_100m.yaml")
    parser.add_argument("--source", default="microsoft/CodeGPT-small-py")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    target = build_model(cfg)
    source = AutoModelForCausalLM.from_pretrained(args.source)
    source_tokenizer = AutoTokenizer.from_pretrained(args.source)
    target_tokenizer = Tokenizer.from_file(args.tokenizer) if args.tokenizer else source_tokenizer.backend_tokenizer
    source_state = source.transformer
    target_state = target.model
    with torch.no_grad():
        # Transfer the vocabulary by encoding each target token's decoded string in CodeGPT's tokenizer.
        source_embed = source_state.wte.weight.detach().float()
        source_lm_head = source.lm_head.weight.detach().float()
        for target_id in range(cfg["model"]["vocab_size"]):
            text = target_tokenizer.decode([target_id])
            ids = source_tokenizer.encode(text, add_special_tokens=False)
            if ids:
                indices = torch.tensor(ids, dtype=torch.long)
                target.model.embed_tokens.weight[target_id].copy_(source_embed[indices].mean(dim=0).to(target.model.embed_tokens.weight.dtype))
                target.lm_head.weight[target_id].copy_(source_lm_head[indices].mean(dim=0).to(target.lm_head.weight.dtype))

        for index, layer in enumerate(target.model.layers):
            base = source_state.h[index]
            qkv = base.attn.c_attn.weight.detach().T
            q, k, v = qkv.chunk(3, dim=0)
            layer.self_attn.q_proj.weight.copy_(q)
            layer.self_attn.k_proj.weight.copy_(k)
            layer.self_attn.v_proj.weight.copy_(v)
            layer.self_attn.o_proj.weight.copy_(base.attn.c_proj.weight.detach().T)
            layer.input_layernorm.weight.copy_(base.ln_1.weight.detach())
            layer.post_attention_layernorm.weight.copy_(base.ln_2.weight.detach())
            # Llama SwiGLU has 1,536-wide projections; seed both branches from GPT-2's first half.
            fc = base.mlp.c_fc.weight.detach().T[: cfg["model"]["intermediate_size"]]
            down = base.mlp.c_proj.weight.detach()[: cfg["model"]["intermediate_size"]].T
            layer.mlp.gate_proj.weight.copy_(fc)
            layer.mlp.up_proj.weight.copy_(fc)
            layer.mlp.down_proj.weight.copy_(down)
        target.model.norm.weight.copy_(source_state.ln_f.weight.detach())
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    target.save_pretrained(output, safe_serialization=True)
    target_tokenizer.save(str(output / "tokenizer.json"))
    (output / "bootstrap.json").write_text(
        '{\n  "source": "' + args.source + '",\n  "parameters": ' + str(count_parameters(target)) +
        ',\n  "context_length": ' + str(cfg["model"]["context_length"]) + '\n}\n', encoding="utf-8"
    )
    print(f"saved {count_parameters(target):,} parameters to {output}")


if __name__ == "__main__":
    main()
