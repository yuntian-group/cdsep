"""Run the synthetic function induction experiments."""

from __future__ import annotations

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.synthetic.agents import INITIAL_PROMPT, NAIVE_INITIAL_PROMPT, SyntheticControl, make_synthetic_agent
from experiments.synthetic.configs import (
    BATCH_SIZE,
    FUNC_NAMES,
    MODEL,
    N_FEW_SHOT,
    N_TEST,
    N_TRAIN,
    OPT_ITERATIONS,
    OPTIMIZER_MODEL,
    SEEDS,
)
from experiments.synthetic.data import (
    format_examples_as_table,
    format_query,
    generate_dataset,
)
from cdsep.episode import run_single_agent_episode
from cdsep.graph import ComputationGraph
from cdsep.llm import LLMClient
from cdsep.logging_utils import ExperimentLogger
from cdsep.optimizer import TextGradOptimizer


def evaluate_agent(agent, test_data, llm, few_shot_text):
    """Evaluate agent on test data. Returns (accuracy, stability, details)."""
    correct = 0
    stable = 0
    details = []

    for ex in test_data:
        query = f"{few_shot_text}\n\n{format_query(ex)}"
        trace = run_single_agent_episode(agent, query, llm)

        predicted = None
        is_stable = trace.is_stable
        if is_stable and trace.steps[0].control is not None:
            predicted = trace.steps[0].control.answer
            stable += 1

        is_correct = predicted == ex["answer"]
        if is_correct:
            correct += 1

        details.append({
            "a": ex["a"], "b": ex["b"],
            "expected": ex["answer"], "predicted": predicted,
            "correct": is_correct, "stable": is_stable,
        })

    n = len(test_data)
    return correct / n if n else 0, stable / n if n else 0, details


def run_single_experiment(func_name: str, method: str, seed: int, logger: ExperimentLogger):
    """Run one experiment: func_name x method x seed."""
    rng = random.Random(seed)
    train_data, test_data = generate_dataset(func_name, N_TRAIN, N_TEST, seed=seed)

    few_shot_indices = rng.sample(range(len(train_data)), min(N_FEW_SHOT, len(train_data)))
    few_shot = [train_data[i] for i in few_shot_indices]
    few_shot_text = format_examples_as_table(few_shot, show_answer=True)

    llm = LLMClient(model=MODEL, temperature=0, logger=logger)

    if method == "fixed":
        agent = make_synthetic_agent(few_shot_text, separated=True)
        acc, stab, details = evaluate_agent(agent, test_data, llm, few_shot_text)
        logger.log_metrics(0, {
            "func": func_name, "method": method, "seed": seed,
            "accuracy": acc, "stability": stab,
        })
        return {"accuracy": acc, "stability": stab, "iterations": [{"k": 0, "accuracy": acc, "stability": stab}]}

    separated = method == "ours"
    opt_llm = LLMClient(model=OPTIMIZER_MODEL, temperature=1, logger=logger)
    optimizer = TextGradOptimizer(opt_llm)

    base_template = INITIAL_PROMPT if separated else NAIVE_INITIAL_PROMPT
    current_prompt = base_template.format(examples=few_shot_text)
    iteration_results = []

    for k in range(OPT_ITERATIONS):
        from cdsep.agent import Agent
        agent = Agent("solver", SyntheticControl, current_prompt, separated=separated)

        batch_indices = rng.sample(range(len(train_data)), min(BATCH_SIZE, len(train_data)))
        batch = [train_data[i] for i in batch_indices]

        graph = ComputationGraph()
        batch_correct = 0
        batch_stable = 0

        for ex in batch:
            query = format_query(ex)
            trace = run_single_agent_episode(agent, query, llm)
            graph.add_from_trace(trace, {"solver": current_prompt})

            if trace.is_stable and trace.steps[0].control is not None:
                batch_stable += 1
                if trace.steps[0].control.answer == ex["answer"]:
                    batch_correct += 1

        batch_acc = batch_correct / len(batch)
        loss = 1.0 - batch_acc

        feedback_lines = [f"Batch accuracy: {batch_acc:.2f} ({batch_correct}/{len(batch)})"]
        for ex in batch:
            feedback_lines.append(f"  f({ex['a']}, {ex['b']}) = {ex['answer']}")

        new_prompt = optimizer.optimize(
            agent, graph, loss,
            feedback="\n".join(feedback_lines),
            separated=separated,
        )

        logger.log_optimization_step(k, "solver", current_prompt, new_prompt, loss)
        current_prompt = new_prompt

        agent_eval = Agent("solver", SyntheticControl, current_prompt, separated=separated)
        acc, stab, _ = evaluate_agent(agent_eval, test_data, llm, few_shot_text)
        logger.log_metrics(k, {
            "func": func_name, "method": method, "seed": seed,
            "iteration": k, "accuracy": acc, "stability": stab,
            "batch_accuracy": batch_acc,
        })
        iteration_results.append({"k": k, "accuracy": acc, "stability": stab})
        print(f"    Iter {k}: test_acc={acc:.3f}, stability={stab:.3f}, batch_acc={batch_acc:.3f}")

    final = iteration_results[-1] if iteration_results else {"accuracy": 0, "stability": 0}
    return {
        "accuracy": final["accuracy"],
        "stability": final["stability"],
        "iterations": iteration_results,
    }


def main():
    os.makedirs("results/synthetic", exist_ok=True)
    all_results = {}

    for func_name in FUNC_NAMES:
        all_results[func_name] = {}
        for method in ["fixed", "naive", "ours"]:
            print(f"\n{'='*60}")
            print(f"Running: {func_name} / {method}")
            print(f"{'='*60}")

            seed_results = []
            for seed in SEEDS:
                print(f"  Seed {seed}:")
                logger = ExperimentLogger("synthetic", f"{func_name}_{method}_s{seed}")
                result = run_single_experiment(func_name, method, seed, logger)
                seed_results.append(result)
                print(f"  -> acc={result['accuracy']:.3f}, stab={result['stability']:.3f}")

            avg_acc = sum(r["accuracy"] for r in seed_results) / len(seed_results)
            avg_stab = sum(r["stability"] for r in seed_results) / len(seed_results)
            all_results[func_name][method] = {
                "mean_accuracy": round(avg_acc, 4),
                "mean_stability": round(avg_stab, 4),
                "per_seed": seed_results,
            }
            print(f"  AVERAGE: acc={avg_acc:.3f}, stab={avg_stab:.3f}")

    with open("results/synthetic/results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\nFINAL SUMMARY")
    print("=" * 70)
    print(f"{'Function':<12} {'Method':<10} {'Accuracy':>10} {'Stability':>10}")
    print("-" * 70)
    for func_name in FUNC_NAMES:
        for method in ["fixed", "naive", "ours"]:
            r = all_results[func_name][method]
            print(f"{func_name:<12} {method:<10} {r['mean_accuracy']:>10.3f} {r['mean_stability']:>10.3f}")

    return all_results


if __name__ == "__main__":
    main()
