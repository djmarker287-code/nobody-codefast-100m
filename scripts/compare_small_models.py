"""Compare the selected model with nearby-size public coding baselines."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TASKS = [
    ('python_add', 'def add_numbers(a, b):\n', ['return']),
    ('python_async', 'async def fetch_json(client, url):\n', ['await', 'return']),
    ('javascript_add', 'function addNumbers(a, b) {\n', ['return', '}']),
    ('javascript_async', 'async function loadJson(url) {\n', ['await', 'fetch', 'return']),
    ('typescript_user', 'type User = {\n', ['id', 'string']),
    ('typescript_handler', 'export async function handler(req: Request): Promise<Response> {\n', ['return', 'Response']),
    ('sql_users', 'SELECT u.id, u.email\nFROM users u\n', ['WHERE']),
    ('html_form', '<form method="post" action="/login">\n', ['input', '</form>']),
    ('css_card', '.card {\n', ['display', 'padding']),
]


def run(name: str, source: str, max_new_tokens: int) -> dict:
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForCausalLM.from_pretrained(source).eval()
    params = sum(p.numel() for p in model.parameters())
    rows = []
    for task, prompt, markers in TASKS:
        inputs = tokenizer(prompt, return_tensors='pt')
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                    pad_token_id=model.config.pad_token_id or model.config.eos_token_id,
                                    eos_token_id=model.config.eos_token_id)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        missing = [m for m in markers if m.lower() not in text.lower()]
        rows.append({'name': task, 'passed': not missing, 'missing': missing, 'generated': text})
    result = {'name': name, 'source': source, 'parameters': params, 'examples': len(rows),
              'passed': sum(r['passed'] for r in rows), 'seconds': round(time.perf_counter() - started, 3),
              'results': rows}
    del model, tokenizer
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    args = parser.parse_args()
    models = [
        ('NOBODY CodeFast-100M', 'packages/nobody-codefast-100m-unsloth'),
        ('CodeParrot-small-multi', 'codeparrot/codeparrot-small-multi'),
        ('CodeGPT-small-py', 'microsoft/CodeGPT-small-py'),
        ('CodeGen-350M-multi reference', 'Salesforce/codegen-350M-multi'),
    ]
    results = [run(name, source, args.max_new_tokens) for name, source in models]
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'tasks': len(TASKS), 'results': results}, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'scores': [(r['name'], r['parameters'], r['passed']) for r in results]}, indent=2))


if __name__ == '__main__':
    main()
