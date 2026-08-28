#!/usr/bin/env python3
"""03 - TextGrad optimization loop.

Trains a single-agent solver on a tiny modular-arithmetic dataset and shows
how accuracy improves over 5 optimization iterations. Uses the *separated*
optimizer mode -- the schema scaffolding stays frozen, so stability remains
100% throughout.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/03_textgrad_optimization.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from cdsep import (
    Agent,
    ComputationGraph,
    LLMClient,
    TextGradOptimizer,
    run_single_agent_episode,
)

K_ITERATIONS = 5
BATCH = 6
TEST_N = 12
SEED = 0


@dataclass
class Answer:
    answer: int


def gen_examples(n: int, rng: random.Random) -> list[dict]:
    """Each pair (a, b) maps to (a + b) mod 7."""
    out = []
    for _ in range(n):
        a, b = rng.randint(0, 12), rng.randint(0, 12)
        out.append({"a": a, "b": b, "answer": (a + b) % 7})
    return out


def evaluate(prompt: str, test: list[dict], llm: LLMClient) -> float:
    agent = Agent("solver", Answer, prompt, separated=True)
    correct = 0
    for ex in test:
        trace = run_single_agent_episode(
            agent, f"f({ex['a']}, {ex['b']}) = ?", llm
        )
        if trace.is_stable and trace.steps[0].control is not None:
            if trace.steps[0].control.answer == ex["answer"]:
                correct += 1
    return correct / len(test)


def main() -> None:
    rng = random.Random(SEED)
    train = gen_examples(40, rng)
    test = gen_examples(TEST_N, rng)

    llm = LLMClient(model="gpt-5.4-nano", temperature=0)
    opt_llm = LLMClient(model="gpt-5.4-mini", temperature=1)
    optimizer = TextGradOptimizer(opt_llm)

    prompt = "You are given examples of f(a, b). Discover the rule and answer."
    print(f"iter 0  test_acc = {evaluate(prompt, test, llm):.3f}  (initial)")

    for k in range(K_ITERATIONS):
        agent = Agent("solver", Answer, prompt, separated=True)
        graph = ComputationGraph()
        batch = rng.sample(train, BATCH)
        correct = 0
        for ex in batch:
            trace = run_single_agent_episode(
                agent, f"f({ex['a']}, {ex['b']}) = ?", llm
            )
            graph.add_from_trace(trace, {"solver": prompt})
            if trace.is_stable and trace.steps[0].control is not None:
                if trace.steps[0].control.answer == ex["answer"]:
                    correct += 1
        loss = 1.0 - correct / BATCH

        feedback = "Batch results:\n" + "\n".join(
            f"  f({ex['a']}, {ex['b']}) = {ex['answer']}" for ex in batch
        )
        prompt = optimizer.optimize(
            agent, graph, loss, feedback=feedback, separated=True
        )

        acc = evaluate(prompt, test, llm)
        print(f"iter {k + 1}  test_acc = {acc:.3f}  loss = {loss:.2f}")


if __name__ == "__main__":
    main()
