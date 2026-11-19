"""Run generated Python against a held-out task set and report syntax/tests."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, StoppingCriteriaList
from generate_checked import ValidPythonStopping


def longest_valid_prefix(source: str) -> str:
    """Trim an invalid continuation while preserving the longest valid Python prefix."""
    lines = source.splitlines()
    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end]).rstrip() + "\n"
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue
    return source


def run_task(source: str, test: str) -> tuple[bool, bool, str | None]:
    try:
        ast.parse(source)
        namespace = {}
        exec(compile(source, "<generated>", "exec"), namespace, namespace)
        return True, bool(eval(test, namespace, namespace)), None
    except Exception as exc:
        return False, False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--early-stop", action="store_true", help="stop when the generated Python becomes valid")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint); tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device).eval()
    results = []
    for line in Path(args.tasks).read_text(encoding="utf-8").splitlines():
        task = json.loads(line); prompt = task["code"].rsplit("\n", 1)[0] + "\n"
        prompt_ids = tokenizer.encode(prompt).ids
        ids = torch.tensor([prompt_ids], device=device)
        stopping = None if not args.early_stop else StoppingCriteriaList([ValidPythonStopping(tokenizer, len(prompt_ids))])
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False, eos_token_id=model.config.eos_token_id, pad_token_id=model.config.pad_token_id, stopping_criteria=stopping)
        generated = tokenizer.decode(out[0].tolist())
        syntax_ok, test_ok, error = run_task(generated, task["test"])
        repaired = longest_valid_prefix(generated) if not syntax_ok else generated
        repaired_syntax_ok, repaired_test_ok, repaired_error = run_task(repaired, task["test"])
        results.append({
            "name": task["name"],
            "syntax_ok": syntax_ok,
            "test_ok": test_ok,
            "error": error,
            "generated": generated,
            "repaired_generated": repaired,
            "repaired_syntax_ok": repaired_syntax_ok,
            "repaired_test_ok": repaired_test_ok,
            "repaired_error": repaired_error,
        })
    report = {
        "examples": len(results),
        "syntax_pass": sum(r["syntax_ok"] for r in results),
        "test_pass": sum(r["test_ok"] for r in results),
        "repaired_syntax_pass": sum(r["repaired_syntax_ok"] for r in results),
        "repaired_test_pass": sum(r["repaired_test_ok"] for r in results),
        "repaired_examples": sum(r["repaired_generated"] != r["generated"] for r in results),
        "results": results,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("examples", "syntax_pass", "test_pass")}, indent=2))


if __name__ == "__main__":
    main()
