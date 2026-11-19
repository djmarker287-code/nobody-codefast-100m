"""Compare direct Transformers output with Unsloth CLI output for the package."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS = [
    'def add_numbers(a, b):',
    'def fibonacci(n):',
    'async def fetch_json(client, url):',
    'function addNumbers(a, b) {',
]


def direct(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                pad_token_id=model.config.pad_token_id,
                                eos_token_id=model.config.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def unsloth(executable: str, package: Path, prompt: str) -> dict:
    started = time.perf_counter()
    result = subprocess.run(
        [executable, 'inference', str(package), prompt, '--max-new-tokens', '64', '--no-server', '--no-load-in-4bit'],
        capture_output=True, text=True, timeout=180,
    )
    output = result.stdout + ('\nSTDERR:\n' + result.stderr if result.stderr else '')
    answer = output.split('Assistant:', 1)[-1] if 'Assistant:' in output else output
    answer = answer.split('Hardware detected:', 1)[0].strip()
    return {'returncode': result.returncode, 'seconds': round(time.perf_counter() - started, 3), 'answer': answer}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', required=True)
    parser.add_argument('--unsloth-exe', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    package = Path(args.package)
    tokenizer = AutoTokenizer.from_pretrained(package)
    model = AutoModelForCausalLM.from_pretrained(package).eval()
    rows = []
    for prompt in PROMPTS:
        rows.append({'prompt': prompt, 'transformers': direct(model, tokenizer, prompt),
                     'unsloth': unsloth(args.unsloth_exe, package, prompt)})
    report = {'package': str(package.resolve()), 'prompts': len(rows), 'results': rows,
              'backend': 'Unsloth Studio CLI / local Radeon backend'}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'prompts': len(rows),
                      'unsloth_successes': sum(r['unsloth']['returncode'] == 0 for r in rows)}, indent=2))


if __name__ == '__main__':
    main()
