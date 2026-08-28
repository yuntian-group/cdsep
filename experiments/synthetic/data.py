"""Generate synthetic function induction datasets."""

from __future__ import annotations

import random
from typing import Callable


FUNCTIONS: dict[str, Callable[[int, int], int]] = {
    "Max": max,
    "Min": min,
    "ModSum10": lambda a, b: (a + b) % 10,
    "Multiply": lambda a, b: a * b,
}


def generate_dataset(
    func_name: str,
    n_train: int = 50,
    n_test: int = 50,
    value_range: tuple[int, int] = (0, 9),
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Generate train and test examples for a synthetic function.

    Each example is {"a": int, "b": int, "answer": int}.
    """
    rng = random.Random(seed)
    func = FUNCTIONS[func_name]
    lo, hi = value_range

    all_pairs = [(a, b) for a in range(lo, hi + 1) for b in range(lo, hi + 1)]
    rng.shuffle(all_pairs)

    examples = []
    for a, b in all_pairs[: n_train + n_test]:
        examples.append({"a": a, "b": b, "answer": func(a, b)})

    return examples[:n_train], examples[n_train : n_train + n_test]


def format_examples_as_table(examples: list[dict], show_answer: bool = True) -> str:
    """Format examples as a text table for the agent prompt."""
    lines = ["| a | b | result |", "|---|---|--------|"]
    for ex in examples:
        if show_answer:
            lines.append(f"| {ex['a']} | {ex['b']} | {ex['answer']} |")
        else:
            lines.append(f"| {ex['a']} | {ex['b']} | ? |")
    return "\n".join(lines)


def format_query(example: dict) -> str:
    return f"What is f({example['a']}, {example['b']})?"
