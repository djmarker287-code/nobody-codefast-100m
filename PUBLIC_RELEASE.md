# Public release checklist

Release name: **NOBODY CodeFast-100M**

This release publishes the training and evaluation workflow while leaving
large local artifacts out of Git. The selected checkpoint is an honest,
measured prototype: strong on the narrow held-out Python gate, weak on the
broader multilingual smoke test, and not yet suitable to describe as perfect.

## Reproduce the local checks

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements.txt

.venv\Scripts\python.exe scripts\evaluate_tasks.py `
  --checkpoint outputs\codefast-appstack-v2\step-1100 `
  --tasks work\quality_eval.jsonl `
  --output outputs\codefast-appstack-v2\quality_eval_public.json

.venv\Scripts\python.exe scripts\evaluate_multilang.py `
  --checkpoint outputs\codefast-appstack-v2\step-1100 `
  --tasks work\multilang_eval.jsonl `
  --output outputs\codefast-appstack-v2\multilang_eval_public.json
```

## Rebuild the package

```powershell
.venv\Scripts\python.exe scripts\package_unsloth.py `
  --checkpoint outputs\codefast-appstack-v2\step-1100 `
  --output packages\nobody-codefast-100m-unsloth

.venv\Scripts\python.exe scripts\test_unsloth_package.py `
  --package packages\nobody-codefast-100m-unsloth
```

## What is deliberately not in Git

`outputs/`, `work/`, and `packages/` are ignored because they contain local
checkpoints, corpora, and archives ranging from hundreds of megabytes to more
than 400 MB. The public repository contains the code and documentation needed
to reproduce them without silently committing machine-specific artifacts.
