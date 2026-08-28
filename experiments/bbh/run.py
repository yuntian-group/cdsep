"""Run BBH (Big-Bench Hard) experiments."""

from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.bbh.agents import (
    NAIVE_PROMPT_TEMPLATE,
    SEP_PROMPT_TEMPLATE,
    make_bbh_agent,
    make_control_schema,
)
from experiments.bbh.configs import (
    BATCH_SIZE,
    MODEL,
    N_FEW_SHOT,
    N_TEST,
    N_TRAIN,
    N_VAL,
    OPT_ITERATIONS,
    OPTIMIZER_MODEL,
    SEEDS,
    TASKS,
)
from experiments.bbh.data import format_few_shot, format_query, generate_dataset, is_correct
from cdsep.agent import Agent
from cdsep.episode import run_single_agent_episode
from cdsep.graph import ComputationGraph
from cdsep.llm import LLMClient
from cdsep.logging_utils import ExperimentLogger
from cdsep.optimizer import TextGradOptimizer


def evaluate_agent(agent, test_data, llm, meta):
    correct = 0
    stable = 0
    details = []
    for ex in test_data:
        trace = run_single_agent_episode(agent, format_query(ex), llm)
        predicted = None
        is_stable = trace.is_stable
        if is_stable and trace.steps[0].control is not None:
            predicted = trace.steps[0].control.answer
            stable += 1
        ok = is_correct(predicted, ex["answer"], meta["kind"])
        if ok:
            correct += 1
        details.append({
            "expected": ex["answer"],
            "predicted": predicted,
            "correct": ok,
            "stable": is_stable,
        })
    n = len(test_data)
    return correct / n if n else 0, stable / n if n else 0, details


def run_single_experiment(task: str, method: str, seed: int, logger: ExperimentLogger):
    rng = random.Random(seed)
    train_data, val_data, test_data, meta = generate_dataset(
        task, N_TRAIN, N_TEST, seed=seed, n_val=N_VAL
    )
    schema_cls = make_control_schema(meta)

    few_shot_indices = rng.sample(range(len(train_data)), min(N_FEW_SHOT, len(train_data)))
    few_shot = [train_data[i] for i in few_shot_indices]
    few_shot_text = format_few_shot(few_shot)

    llm = LLMClient(model=MODEL, temperature=0, logger=logger)

    if method == "fixed":
        agent = make_bbh_agent(few_shot_text, meta, separated=True)
        acc, stab, _ = evaluate_agent(agent, test_data, llm, meta)
        logger.log_metrics(0, {
            "task": task, "method": method, "seed": seed,
            "accuracy": acc, "stability": stab,
        })
        return {
            "accuracy": acc, "stability": stab,
            "iterations": [{"k": 0, "accuracy": acc, "stability": stab}],
        }

    separated = method == "ours"
    opt_llm = LLMClient(model=OPTIMIZER_MODEL, temperature=1, logger=logger)
    optimizer = TextGradOptimizer(opt_llm)

    template = SEP_PROMPT_TEMPLATE if separated else NAIVE_PROMPT_TEMPLATE
    current_prompt = template.format(few_shot=few_shot_text)
    iteration_results = []

    for k in range(OPT_ITERATIONS):
        agent = Agent("solver", schema_cls, current_prompt, separated=separated)

        batch_indices = rng.sample(range(len(train_data)), min(BATCH_SIZE, len(train_data)))
        batch = [train_data[i] for i in batch_indices]

        graph = ComputationGraph()
        batch_correct = 0
        for ex in batch:
            trace = run_single_agent_episode(agent, format_query(ex), llm)
            graph.add_from_trace(trace, {"solver": current_prompt})
            if trace.is_stable and trace.steps[0].control is not None:
                if is_correct(trace.steps[0].control.answer, ex["answer"], meta["kind"]):
                    batch_correct += 1

        batch_acc = batch_correct / len(batch)
        loss = 1.0 - batch_acc
        feedback_lines = [f"Batch accuracy: {batch_acc:.2f} ({batch_correct}/{len(batch)})"]
        for ex in batch:
            feedback_lines.append(f"  expected answer: {ex['answer']}")

        new_prompt = optimizer.optimize(
            agent, graph, loss,
            feedback="\n".join(feedback_lines),
            separated=separated,
        )
        logger.log_optimization_step(k, "solver", current_prompt, new_prompt, loss)
        current_prompt = new_prompt

        eval_agent = Agent("solver", schema_cls, current_prompt, separated=separated)
        acc, stab, _ = evaluate_agent(eval_agent, test_data, llm, meta)
        val_acc, val_stab = evaluate_agent(eval_agent, val_data, llm, meta)
        logger.log_metrics(k, {
            "task": task, "method": method, "seed": seed,
            "iteration": k, "accuracy": acc, "stability": stab,
            "val_accuracy": val_acc,
            "batch_accuracy": batch_acc,
        })
        iteration_results.append({"k": k, "accuracy": acc, "stability": stab, "val_accuracy": val_acc})
        print(f"    Iter {k}: test_acc={acc:.3f} val_acc={val_acc:.3f}", flush=True)

    from experiments.common.splits import pick_best_iteration
    best = pick_best_iteration(iteration_results, "val_accuracy")
    return {
        "accuracy": best.get("accuracy", 0),
        "stability": best.get("stability", 0),
        "best_iter": best.get("k", 0),
        "iterations": iteration_results,
    }


def main():
    os.makedirs("results/bbh", exist_ok=True)
    all_results = {}

    for task in TASKS:
        all_results[task] = {}
        for method in ["fixed", "naive", "ours"]:
            print(f"\n{'='*60}")
            print(f"Running: {task} / {method}")
            print(f"{'='*60}", flush=True)

            seed_results = []
            for seed in SEEDS:
                print(f"  Seed {seed}:", flush=True)
                logger = ExperimentLogger("bbh", f"{task}_{method}_s{seed}")
                result = run_single_experiment(task, method, seed, logger)
                seed_results.append(result)
                print(f"  -> acc={result['accuracy']:.3f}, stab={result['stability']:.3f}", flush=True)

            avg_acc = sum(r["accuracy"] for r in seed_results) / len(seed_results)
            avg_stab = sum(r["stability"] for r in seed_results) / len(seed_results)
            all_results[task][method] = {
                "mean_accuracy": round(avg_acc, 4),
                "mean_stability": round(avg_stab, 4),
                "per_seed": seed_results,
            }
            print(f"  AVERAGE: acc={avg_acc:.3f}, stab={avg_stab:.3f}", flush=True)

    with open("results/bbh/results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\nBBH SUMMARY")
    print("=" * 80)
    print(f"{'Task':<45} {'Method':<10} {'Accuracy':>10} {'Stability':>10}")
    print("-" * 80)
    for task in TASKS:
        for method in ["fixed", "naive", "ours"]:
            r = all_results[task][method]
            print(f"{task:<45} {method:<10} {r['mean_accuracy']:>10.3f} {r['mean_stability']:>10.3f}")


if __name__ == "__main__":
    main()
