"""Create a ~95M GPT-NeoX student with GPT-2/CodeGPT-compatible weights."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoXConfig, GPTNeoXForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='configs/code_100m_neox_codegpt.yaml')
    parser.add_argument('--source', default='microsoft/CodeGPT-small-py')
    parser.add_argument('--tokenizer', required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))['model']
    target_cfg = GPTNeoXConfig(
        vocab_size=cfg['vocab_size'],
        hidden_size=cfg['hidden_size'],
        num_hidden_layers=cfg['num_layers'],
        num_attention_heads=cfg['num_heads'],
        intermediate_size=cfg['intermediate_size'],
        max_position_embeddings=cfg['context_length'],
        rotary_pct=cfg['rotary_pct'],
        rotary_emb_base=cfg['rope_theta'],
        hidden_act=cfg['hidden_act'],
        use_parallel_residual=cfg['use_parallel_residual'],
        hidden_dropout=cfg['dropout'],
        attention_dropout=cfg['dropout'],
        bos_token_id=cfg['bos_token_id'],
        eos_token_id=cfg['eos_token_id'],
        pad_token_id=cfg['pad_token_id'],
        tie_word_embeddings=cfg['tie_word_embeddings'],
    )
    target = GPTNeoXForCausalLM(target_cfg)
    source = AutoModelForCausalLM.from_pretrained(args.source)
    source_tokenizer = AutoTokenizer.from_pretrained(args.source)
    source_state = source.transformer
    target_state = target.gpt_neox
    with torch.no_grad():
        rows = min(source_state.wte.weight.shape[0], target_state.embed_in.weight.shape[0])
        target_state.embed_in.weight[:rows].copy_(source_state.wte.weight[:rows])
        for index, layer in enumerate(target_state.layers):
            base = source_state.h[index]
            layer.input_layernorm.weight.copy_(base.ln_1.weight)
            layer.input_layernorm.bias.copy_(base.ln_1.bias)
            layer.post_attention_layernorm.weight.copy_(base.ln_2.weight)
            layer.post_attention_layernorm.bias.copy_(base.ln_2.bias)
            layer.attention.query_key_value.weight.copy_(base.attn.c_attn.weight.T)
            layer.attention.query_key_value.bias.copy_(base.attn.c_attn.bias)
            layer.attention.dense.weight.copy_(base.attn.c_proj.weight.T)
            layer.attention.dense.bias.copy_(base.attn.c_proj.bias)
            layer.mlp.dense_h_to_4h.weight.copy_(base.mlp.c_fc.weight.T)
            layer.mlp.dense_h_to_4h.bias.copy_(base.mlp.c_fc.bias)
            layer.mlp.dense_4h_to_h.weight.copy_(base.mlp.c_proj.weight.T)
            layer.mlp.dense_4h_to_h.bias.copy_(base.mlp.c_proj.bias)
        target_state.final_layer_norm.weight.copy_(source_state.ln_f.weight)
        target_state.final_layer_norm.bias.copy_(source_state.ln_f.bias)
        if not target.config.tie_word_embeddings:
            target.embed_out.weight[:rows].copy_(source.lm_head.weight[:rows])
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    target.save_pretrained(output, safe_serialization=True)
    Tokenizer.from_file(args.tokenizer).save(str(output / 'tokenizer.json'))
    params = sum(p.numel() for p in target.parameters())
    (output / 'bootstrap.json').write_text(
        '{\n  "source": "' + args.source + '",\n  "parameters": ' + str(params) +
        ',\n  "context_length": ' + str(cfg['context_length']) + '\n}\n', encoding='utf-8')
    print(f'saved {params:,} parameters to {output}; source tokenizer={source_tokenizer.vocab_size}')


if __name__ == '__main__':
    main()
