"""Generate a small teacher-completion corpus from CodeGen-350M-multi."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def source_rows(path: Path):
    with path.open(encoding='utf-8', errors='ignore') as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get('text') or row.get('content') or row.get('code')
            language = str(row.get('language', '')).lower()
            if isinstance(text, str) and language in {'python', 'javascript', 'typescript', 'sql', 'html', 'css'}:
                lines = text.strip().splitlines()
                if len(lines) >= 5:
                    yield language, lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--max-new-tokens', type=int, default=96)
    parser.add_argument('--seed', type=int, default=1337)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    candidates = list(source_rows(Path(args.source_data))); rng.shuffle(candidates)
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codegen-350M-multi')
    model = AutoModelForCausalLM.from_pretrained('Salesforce/codegen-350M-multi').eval()
    rows = []
    for language, lines in candidates:
        if len(rows) >= args.count:
            break
        cut = rng.randint(max(2, len(lines) // 3), max(3, len(lines) - 2))
        prompt = '\n'.join(lines[:cut]).rstrip() + '\n'
        if len(prompt) < 40 or len(prompt) > 1800:
            continue
        inputs = tokenizer(prompt, return_tensors='pt')
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
        completion = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        text = prompt + completion
        if len(completion.strip()) >= 20:
            rows.append({'text': text, 'language': language, 'teacher': 'Salesforce/codegen-350M-multi'})
            if len(rows) % 10 == 0:
                print(f'generated {len(rows)}/{args.count}', flush=True)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps({'output': str(output), 'rows': len(rows)}, indent=2))


if __name__ == '__main__':
    main()
