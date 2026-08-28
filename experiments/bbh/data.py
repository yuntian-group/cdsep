"""Load BBH (Big-Bench Hard) tasks via HuggingFace datasets."""

from __future__ import annotations

import random
import re
from functools import lru_cache
from typing import Any

from datasets import load_dataset


def _parse_options(input_text: str) -> list[str]:
    """Extract option letters [A, B, C, ...] from a BBH input that lists options."""
    matches = re.findall(r"\(([A-Z])\)", input_text)
    seen = set()
    out: list[str] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _normalise_target(target: str) -> str:
    """Convert raw BBH target into canonical answer string.

    For multiple-choice tasks, returns just the letter (e.g. "A").
    For Yes/No tasks, returns "Yes" or "No" verbatim.
    For free-form tasks, returns the raw string.
    """
    s = target.strip()
    m = re.match(r"^\(?([A-Z])\)?$", s)
    if m:
        return m.group(1)
    return s


@lru_cache(maxsize=None)
def _load_raw(task_name: str) -> list[dict]:
    ds = load_dataset("lukaemon/bbh", task_name, split="test")
    return [{"input": ex["input"], "target": ex["target"]} for ex in ds]


def task_answer_space(task_name: str, examples: list[dict]) -> tuple[str, list[str] | None]:
    """Return (answer_kind, allowed_values_or_None).

    answer_kind in {"choice", "yesno", "freeform"}.
    """
    if task_name == "causal_judgement":
        return "yesno", ["Yes", "No"]
    if task_name == "word_sorting":
        return "freeform", None
    # multiple choice: infer letters from first example's options
    options = _parse_options(examples[0]["input"])
    return "choice", options


def generate_dataset(
    task_name: str,
    n_train: int = 25,
    n_test: int = 25,
    seed: int = 42,
    n_val: int = 0,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Return (train, val, test, meta) where each example is {"input": str, "answer": str}.

    meta contains {"task": task_name, "kind": "choice"/"yesno"/"freeform",
                   "options": [...] or None}.
    """
    rng = random.Random(seed)
    raw = list(_load_raw(task_name))
    rng.shuffle(raw)

    examples: list[dict] = []
    for ex in raw:
        examples.append({
            "input": ex["input"],
            "answer": _normalise_target(ex["target"]),
        })

    n = min(n_train + n_val + n_test, len(examples))
    train = examples[:n_train]
    val = examples[n_train : n_train + n_val]
    test = examples[n_train + n_val : n]

    kind, options = task_answer_space(task_name, train)
    meta = {"task": task_name, "kind": kind, "options": options}
    return train, val, test, meta


def format_few_shot(examples: list[dict]) -> str:
    """Format a small set of examples as a text block for use in the prompt."""
    blocks = []
    for ex in examples:
        blocks.append(f"Q: {ex['input']}\nA: {ex['answer']}")
    return "\n\n".join(blocks)


def format_query(example: dict) -> str:
    return example["input"]


def is_correct(predicted: Any, gold: str, kind: str) -> bool:
    """Compare predicted answer to gold under task-appropriate normalisation."""
    if predicted is None:
        return False
    p = str(predicted).strip()
    if kind == "choice":
        m = re.search(r"[A-Z]", p)
        if m:
            p = m.group(0)
        return p.upper() == gold.upper()
    if kind == "yesno":
        pl = p.lower()
        if pl.startswith("y"):
            return gold.lower().startswith("y")
        if pl.startswith("n"):
            return gold.lower().startswith("n")
        return False
    # free-form
    return p.strip().lower() == gold.strip().lower()
