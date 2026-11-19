"""Prepare a GGUF-export source package with recognized CodeGPT BPE files."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(source, output)
    tokenizer = AutoTokenizer.from_pretrained('microsoft/CodeGPT-small-py')
    tokenizer.save_pretrained(output)
    (output / 'gguf_package.json').write_text(json.dumps({
        'source': str(source.resolve()), 'tokenizer_source': 'microsoft/CodeGPT-small-py',
        'purpose': 'Unsloth GGUF export source', 'public': False,
    }, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'files': len(list(output.iterdir()))}, indent=2))


if __name__ == '__main__':
    main()
