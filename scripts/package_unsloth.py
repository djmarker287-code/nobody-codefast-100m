"""Package the selected CodeFast checkpoint for Unsloth Studio and CLI use."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    required = ['config.json', 'generation_config.json', 'model.safetensors', 'tokenizer.json']
    for name in required:
        path = source / name
        if not path.exists():
            raise FileNotFoundError(path)
        shutil.copy2(path, output / name)
    metadata = {
        'name': 'NOBODY CodeFast-100M',
        'status': 'local-test-package',
        'parameters': 99_756_288,
        'context_length': 65_536,
        'architecture': 'LlamaForCausalLM',
        'source_checkpoint': str(source),
        'unsloth_load': {
            'recommended_max_seq_length': 2048,
            'long_context_max_seq_length': 65536,
            'load_in_4bit': False,
            'multi_gpu': False,
        },
        'validation': {
            'python_executable_tasks': '60/60',
            'multilingual_smoke': '4/9',
            'cpu_generation_tokens_per_second': 77.95,
        },
        'publication': {'github_public': False, 'requires_user_approval': True},
    }
    (output / 'unsloth_package.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    tokenizer_config = {
        'tokenizer_class': 'PreTrainedTokenizerFast',
        'fast_tokenizer_file': 'tokenizer.json',
        'model_max_length': 65536,
        'bos_token': '<s>',
        'eos_token': '</s>',
        'pad_token': '<pad>',
        'unk_token': '<|UNKNOWN|>',
        'add_bos_token': False,
        'add_eos_token': False,
        'clean_up_tokenization_spaces': False,
        'model_input_names': ['input_ids', 'attention_mask'],
        'chat_template': "{% for message in messages %}{{ message['content'] }}{% if not loop.last %}\\n{% endif %}{% endfor %}{% if add_generation_prompt %}\\n{% endif %}",
    }
    (output / 'tokenizer_config.json').write_text(json.dumps(tokenizer_config, indent=2), encoding='utf-8')
    (output / 'special_tokens_map.json').write_text(json.dumps({
        'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>', 'unk_token': '<|UNKNOWN|>'
    }, indent=2), encoding='utf-8')
    readme = '''# NOBODY CodeFast-100M\n\nLocal Unsloth package; not published.\n\n## Unsloth CLI\n\n```powershell\nC:\\Users\\Marco\\.unsloth\\studio\\bin\\unsloth.exe inference . --prompt "def add_numbers(a, b):" --max-new-tokens 64 --no-server\n```\n\nFor the 64k context setting, pass `--max-seq-length 65536`. Use 2048 for ordinary fast inference.\n\nThis package is the selected 99,756,288-parameter Transformers checkpoint. It has not been published to GitHub.\n\nGGUF export is currently not included because the bundled llama.cpp converter does not recognize this custom BPE tokenizer.\n'''
    (output / 'README_UNSLOTH.md').write_text(readme, encoding='utf-8')
    print(json.dumps({'output': str(output), 'files': len(list(output.iterdir())), 'parameters': metadata['parameters']}, indent=2))


if __name__ == '__main__':
    main()
