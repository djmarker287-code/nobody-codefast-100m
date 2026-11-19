# NOBODY CodeFast-100M

NOBODY CodeFast-100M is an experimental, approximately 100-million-parameter
decoder-only code model with a 65,536-token context window. The name is a
provisional release name for this public experiment; it does not imply that the
model is production-ready.

## What is included

- A reproducible CPU/GPU-capable training script with resumable checkpoints.
- Dataset preparation helpers for code and mixed application-stack data.
- Held-out executable Python evaluation and a small multilingual smoke test.
- Native Transformers/Unsloth packaging helpers.
- A router that can select the Python or web-specialist checkpoint.

The public repository intentionally does not include checkpoint weights,
generated corpora, or the local Unsloth zip. Those artifacts are large and are
ignored by Git. They can be regenerated locally using the scripts in this repo.

## Selected checkpoint

The selected local checkpoint is `outputs/codefast-appstack-v2/step-1100`:

- 99,756,288 parameters
- 65,536-token context
- 60/60 on the held-out Python syntax and executable-test gate
- 2/9 on the multilingual smoke gate
- approximately 77.95 new tokens/second for the local short-prompt CPU benchmark

The Python gate is narrow and should not be treated as general coding ability.
An unseen free-form Fibonacci prompt was not reliably valid Python, and the
Unsloth interactive test did not produce usable code. This is therefore an
experimental checkpoint, not a claim of perfection.

## Comparison

The same nine-prompt greedy smoke set produced:

| Model | Parameters | Passed |
| --- | ---: | ---: |
| NOBODY CodeFast-100M | 99,756,288 | 2/9 |
| CodeParrot-small-multi | 111,008,256 | 2/9 |
| CodeGPT-small-py | 124,243,200 | 1/9 |
| CodeGen-350M-multi | 356,712,448 | 4/9 |

These are local smoke-test results, not a standardized benchmark or a claim
that the models were trained under identical conditions.

## Limitations

- Training and evaluation were CPU-heavy because the Windows ROCm/HIP runtime
  crashed during GPU backpropagation on this machine.
- The current model needs broader, licensed data and substantially more compute
  for robust multilingual coding quality.
- GGUF export was not available because the bundled converter rejected the
  custom byte-level BPE pre-tokenizer.
- Generated code must be reviewed and tested by a human before use.

## Responsible use

Do not use generated code without review, tests, dependency checks, and normal
security practices. Do not train on source code or data unless its license and
privacy terms permit the intended use.
