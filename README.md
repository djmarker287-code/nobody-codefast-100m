# NOBODY CodeFast-100M

> Experimental public release: a reproducible 100M-parameter coding-model
> workflow with a 65k context window. It is not yet a production-quality or
> perfect model.

See [MODEL_CARD.md](MODEL_CARD.md) for measured results, limitations, and the
comparison against nearby open models. See [PUBLIC_RELEASE.md](PUBLIC_RELEASE.md)
for the public reproducibility checklist.

This project trains a decoder-only causal language model designed for fast, detailed code generation.
The selected architecture is intentionally near 100M: 99,756,288 trainable parameters with a 50,002-token
CodeGPT-compatible byte-level BPE vocabulary, 8 rotary-attention transformer blocks, 768 hidden dimensions,
12 attention heads, and a 65,536-token context window. Rotary positions provide 64k context without adding
a learned position-embedding table.

## Step-by-step: use the model

These commands are for Windows PowerShell. The public repository contains the
code, but not the very large local checkpoint files. On the training computer,
the selected checkpoint is `outputs\codefast-appstack-v2\step-1100`.

1. **Clone the repository and enter it.**

   ```powershell
   git clone https://github.com/djmarker287-code/nobody-codefast-100m.git
   cd nobody-codefast-100m
   ```

2. **Create the Python environment.** Python 3.12 is recommended.

   ```powershell
   uv venv .venv --python 3.12
   uv pip install --python .venv\Scripts\python.exe -r requirements.txt
   ```

3. **Run a normal coding prompt.** Replace the checkpoint path if your model
   files are stored somewhere else.

   ```powershell
   .venv\Scripts\python.exe scripts\sample.py `
     --checkpoint outputs\codefast-appstack-v2\step-1100 `
     --prompt "def add_numbers(a, b):" `
     --max-new-tokens 120
   ```

4. **Use the checked Python mode when you want a syntax safety check.** It
   returns `valid_python: false` instead of pretending malformed output is
   valid code.

   ```powershell
   .venv\Scripts\python.exe scripts\generate_checked.py `
     --checkpoint outputs\codefast-appstack-v2\step-1100 `
     --prompt "def fibonacci(n):"
   ```

5. **Use the web route for JavaScript, TypeScript, SQL, HTML, or CSS.**

   ```powershell
   .venv\Scripts\python.exe scripts\run_codefast.py `
     --checkpoint outputs\codefast-appstack-v2\step-1100 `
     --web-checkpoint outputs\codefast-web-v2\final `
     --prompt "function addNumbers(a, b) {"
   ```

6. **Always review and test the result.** This is an experimental small model;
   do not run generated code against real data or production systems without
   human review, tests, dependency checks, and security checks.

### Is ROCm right for your computer?

ROCm is AMD's GPU software stack. On your computer it is partially useful:
Unsloth detected the Radeon AI PRO R9700 and GPU inference worked, but the
installed Windows ROCm/HIP runtime crashed during training backpropagation.
The older integrated Radeon was excluded because the installed PyTorch build
did not contain compatible kernels.

For now, use ROCm/Unsloth for experimentation and inference, but use the safe
CPU fallback for training:

```powershell
C:\Users\Marco\.unsloth\studio\unsloth_studio\Scripts\python.exe `
  scripts\train_unsloth.py `
  --data work\code.jsonl --language Python `
  --out outputs\unsloth-run --steps 100 --seq-len 1024 --cpu
```

Longer GPU training becomes sensible after moving to a known-compatible
ROCm/PyTorch/Unsloth combination, usually on a supported Linux setup or after
the Windows HIP issue is resolved. You do not need to reinstall ROCm just to
run the model on CPU.

## Quick start

Use Python 3.12 and install the pinned runtime:

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Create a tiny local corpus and run a verification training pass:

```powershell
.venv\Scripts\python.exe scripts\make_demo_data.py --output work\demo.jsonl
.venv\Scripts\python.exe scripts\train.py --data work\demo.jsonl --out outputs\demo --steps 100 --smoke --seq-len 1024
.venv\Scripts\python.exe scripts\sample.py --checkpoint outputs\demo\final --prompt "def fibonacci(n):"
```

The Unsloth Studio environment can run the same Llama checkpoint on its configured accelerator:

```powershell
C:\Users\Marco\.unsloth\studio\unsloth_studio\Scripts\python.exe scripts\train_unsloth.py `
  --data work\code.jsonl --language Python --out outputs\unsloth-run --steps 100 --seq-len 1024
```

On this Windows/ROCm machine, the accelerator currently crashes inside the installed HIP DLL. Use
`--cpu` for a safe fallback until the Unsloth/ROCm runtime is repaired; omit it on a working Unsloth GPU.

The checkpoint supports 65,536-token inference. Use `--seq-len 65536` for long-context training when
GPU memory allows it; the 1,024-token smoke setting only makes verification affordable.

For real training, provide a large JSONL corpus with a `text` field:

```powershell
.venv\Scripts\python.exe scripts\train.py `
  --data work\code.jsonl `
  --language python `
  --out outputs\codefast-100m `
  --tokens 2000000000 `
  --batch-size 4 `
  --grad-accum 16
```

The trainer streams data, packs examples into fixed-length sequences, saves resumable checkpoints,
tracks validation loss/perplexity, and writes a `model_card.json` containing the exact parameter count.
For a useful coding model, plan on hundreds of millions to several billion high-quality code tokens;
the smoke command is only a correctness check, not meaningful pretraining.

## Data format

Local JSONL files may use `text`, `content`, or `code`. One document per line is recommended. The
loader removes obvious secrets and oversized/binary-looking documents, preserves language metadata,
and can filter by `--language`.

## Design choices

- Byte-level BPE makes the model usable for arbitrary source code without an external tokenizer.
- Llama-style RMSNorm plus rotary attention supports the requested 64k context and standard Transformers
  export.
- AdamW, cosine decay, warmup, gradient clipping, bf16/fp16 autocast, and gradient accumulation are
  included for single-GPU or CPU execution.
- A deterministic held-out split, code-generation samples, and a lightweight syntax check are emitted
  during evaluation.

The project does not claim that a short local run produces a production-grade model. The data volume,
hardware, and wall-clock time must be sufficient for the requested training budget.

For a CodeGPT-compatible vocabulary experiment, use `configs/code_100m_codegpt_vocab.yaml` and
`scripts/bootstrap_codegpt.py` before continuing with `scripts/train.py --resume`.

To build a corpus from local Git repositories, use `scripts/build_corpus.py`; this avoids gated dataset
access and lets you review each repository's license before training.

Run the syntax evaluator against a checkpoint with `scripts/eval_codegen.py`.

## Latest quality checkpoint

The selected trained native-CodeGPT-vocabulary checkpoint is in `outputs/codefast-appstack-v2/step-1100`.
It has 99,756,288 parameters and a 65,536-token context window. Run the held-out executable
validation suite with:

```powershell
.venv\Scripts\python.exe scripts\evaluate_tasks.py `
  --checkpoint outputs\codefast-appstack-v2\step-1100 `
  --tasks work\quality_eval.jsonl `
  --output outputs\codefast-appstack-v2\quality_eval_step1100_early.json `
  --early-stop
```

This stops as soon as the generated Python is complete; it scored 60/60 on the held-out executable tasks.

```powershell
.venv\Scripts\python.exe scripts\evaluate_tasks.py `
  --checkpoint outputs\codefast-appstack-v2\step-1100 `
  --tasks work\quality_eval.jsonl `
  --output outputs\codefast-appstack-v2\quality_eval_raw.json
```

For generated Python, `scripts/generate_checked.py` keeps the longest syntactically valid prefix
and returns `valid_python: false` instead of presenting an invalid continuation as checked output:

```powershell
.venv\Scripts\python.exe scripts\generate_checked.py `
  --checkpoint outputs\codefast-appstack-v2\step-1100 `
  --prompt "def factorial_1995(n):"
```

For mixed app work, `scripts/run_codefast.py` routes Python to the full-stack checkpoint and
obvious JavaScript/TypeScript/SQL/HTML/CSS prompts to the web-specialized checkpoint:

```powershell
.venv\Scripts\python.exe scripts\run_codefast.py `
  --checkpoint outputs\codefast-appstack-v2\step-1100 `
  --web-checkpoint outputs\codefast-web-v2\final `
  --prompt "function addNumbers(a, b) {"
```
