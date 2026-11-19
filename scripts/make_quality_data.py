"""Create deterministic coding-practice data and a held-out executable benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def examples():
    for i in range(2000):
        a, b, c = i % 17, (i * 3) % 19, (i * 7) % 23
        yield f"def add_numbers_{i}(a, b):\n    return a + b\n", f"add_numbers_{i}", f"add_numbers_{i}({a}, {b}) == {a + b}"
        yield f"def clamp_{i}(value, low, high):\n    return max(low, min(high, value))\n", f"clamp_{i}", f"clamp_{i}({a - 20}, 0, 10) == 0"
        yield f"def is_even_{i}(value):\n    return value % 2 == 0\n", f"is_even_{i}", f"is_even_{i}({i}) == {i % 2 == 0}"
        yield f"def reverse_words_{i}(text):\n    return ' '.join(reversed(text.split()))\n", f"reverse_words_{i}", f"reverse_words_{i}('one two three') == 'three two one'"
        yield f"def unique_sorted_{i}(items):\n    return sorted(set(items))\n", f"unique_sorted_{i}", f"unique_sorted_{i}([3, 1, 3, 2]) == [1, 2, 3]"
        yield f"def flatten_{i}(groups):\n    return [item for group in groups for item in group]\n", f"flatten_{i}", f"flatten_{i}([[1, 2], [3]]) == [1, 2, 3]"
        yield f"def count_chars_{i}(text):\n    counts = {{}}\n    for char in text:\n        counts[char] = counts.get(char, 0) + 1\n    return counts\n", f"count_chars_{i}", f"count_chars_{i}('aba') == {{'a': 2, 'b': 1}}"
        yield f"def factorial_{i}(n):\n    result = 1\n    for value in range(2, n + 1):\n        result *= value\n    return result\n", f"factorial_{i}", f"factorial_{i}(5) == 120"
        yield f"def fibonacci_{i}(n):\n    first, second = 0, 1\n    for _ in range(n):\n        first, second = second, first + second\n    return first\n", f"fibonacci_{i}", f"fibonacci_{i}(8) == 21"
        yield f"def chunked_{i}(items, size):\n    return [items[index:index + size] for index in range(0, len(items), size)]\n", f"chunked_{i}", f"chunked_{i}([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]"
        yield f"class Counter_{i}:\n    def __init__(self):\n        self.value = 0\n\n    def increment(self, amount=1):\n        self.value += amount\n        return self.value\n", f"Counter_{i}", f"Counter_{i}().increment(3) == 3"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--eval-count", type=int, default=60)
    args = parser.parse_args()
    rows = list(examples())
    train_path, eval_path = Path(args.train), Path(args.eval)
    train_path.parent.mkdir(parents=True, exist_ok=True); eval_path.parent.mkdir(parents=True, exist_ok=True)
    held_out = min(args.eval_count, len(rows))
    with train_path.open("w", encoding="utf-8") as train_handle:
        for code, _, _ in rows[:-held_out]:
            train_handle.write(json.dumps({"text": code, "language": "Python"}) + "\n")
    with eval_path.open("w", encoding="utf-8") as eval_handle:
        for code, name, test in rows[-held_out:]:
            eval_handle.write(json.dumps({"code": code, "name": name, "test": test, "language": "Python"}) + "\n")
    print(f"wrote {len(rows) - held_out} training examples and {held_out} held-out evaluation tasks")


if __name__ == "__main__":
    main()
