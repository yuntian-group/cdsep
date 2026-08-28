"""Run the medical-impairment underwriting experiment.

This is the medical-impairment version of the synthetic underwriting
experiment. The pipeline is a three-agent sequential program:
``medical_extractor`` -> ``impairment_rater`` -> ``final_aggregator``.
All control-routing fields are closed ``Literal`` types so the
optimizer cannot break protocol no matter how it edits the prompts.
"""

from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from experiments.insurance.agents import (
    FINAL_AGGREGATOR_PROMPT,
    IMPAIRMENT_RATER_PROMPT,
    MEDICAL_EXTRACTOR_PROMPT,
    NAIVE_FINAL_AGGREGATOR_PROMPT,
    NAIVE_IMPAIRMENT_RATER_PROMPT,
    NAIVE_MEDICAL_EXTRACTOR_PROMPT,
    FinalAggregatorControl,
    ImpairmentRaterControl,
    MedicalExtractorControl,
    make_insurance_agents,
)
from experiments.insurance.configs import (
    BATCH_SIZE,
    MAX_ORDINAL_DISTANCE,
    MODEL,
    N_SAMPLES,
    N_TRAIN,
    N_VAL,
    OPT_ITERATIONS,
    OPTIMIZER_MODEL,
    RATING_TO_ORDINAL,
    SEEDS,
)
from experiments.insurance.data import generate_dataset
from cdsep.agent import Agent
from cdsep.episode import EpisodeTrace, run_episode
from cdsep.graph import ComputationGraph
from cdsep.llm import LLMClient
from cdsep.logging_utils import ExperimentLogger
from cdsep.optimizer import TextGradOptimizer


def route_insurance(control) -> str:
    """Strict sequential routing: extractor -> rater -> aggregator -> done."""
    if isinstance(control, MedicalExtractorControl):
        return "impairment_rater"
    if isinstance(control, ImpairmentRaterControl):
        return "final_aggregator"
    if isinstance(control, FinalAggregatorControl):
        return "terminate"
    return "terminate"


def run_insurance_episode(
    agents: dict[str, Agent], applicant: dict, llm: LLMClient,
) -> tuple[EpisodeTrace, str | None]:
    trace = run_episode(
        entry_agent=agents["medical_extractor"],
        agents=agents,
        route_fn=route_insurance,
        task_input=applicant["description"],
        llm=llm,
        max_steps=8,
    )
    predicted = None
    if trace.completed_cleanly:
        last = trace.steps[-1]
        if hasattr(last.control, "final_rating"):
            predicted = last.control.final_rating
    return trace, predicted


def evaluate_insurance(
    agents: dict[str, Agent], test_data: list[dict], llm: LLMClient,
) -> dict:
    import os
    from concurrent.futures import ThreadPoolExecutor

    def _one(applicant):
        trace, predicted = run_insurance_episode(agents, applicant, llm)
        gt = applicant["ground_truth"]
        is_stable = trace.is_stable
        ok = predicted == gt
        if predicted in RATING_TO_ORDINAL and gt in RATING_TO_ORDINAL:
            ae = abs(RATING_TO_ORDINAL[predicted] - RATING_TO_ORDINAL[gt])
        else:
            ae = MAX_ORDINAL_DISTANCE
        return is_stable, ok, ae, {
            "ground_truth": gt, "predicted": predicted,
            "correct": ok, "stable": is_stable, "ae": ae,
        }

    n = len(test_data)
    workers = max(1, min(int(os.environ.get("INS_EVAL_WORKERS", "12")), n)) if n else 1
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_one, test_data))
    stable = sum(1 for r in results if r[0])
    correct = sum(1 for r in results if r[1])
    total_ae = sum(r[2] for r in results)
    details = [r[3] for r in results]
    n = len(test_data)
    return {
        "accuracy": correct / n if n else 0.0,
        "mae": total_ae / n if n else 0.0,
        "stability": stable / n if n else 0.0,
        "details": details,
    }


def run_insurance_experiment(
    method: str,
    seed: int,
    logger: ExperimentLogger,
    *,
    agent_model: str | None = None,
    optimizer_model: str | None = None,
    feedback_mode: str = "scalar",
    max_parse_retries: int | None = None,
) -> dict:
    train_data, val_data, test_data = generate_dataset(
        N_SAMPLES, seed=seed, n_train=N_TRAIN, n_val=N_VAL
    )
    rng = random.Random(seed)
    separated = method == "ours"

    agent_model = agent_model or MODEL
    optimizer_model = optimizer_model or OPTIMIZER_MODEL
    llm = LLMClient(model=agent_model, temperature=0, logger=logger)

    def _apply_retries(agents):
        if max_parse_retries is not None:
            for a in agents.values():
                a.max_parse_retries = max_parse_retries
        return agents

    if method == "fixed":
        agents = _apply_retries(make_insurance_agents(separated=True))
        metrics = evaluate_insurance(agents, test_data, llm)
        logger.log_metrics(0, {"method": "fixed", "seed": seed,
                               **{k: v for k, v in metrics.items() if k != "details"}})
        return {
            "accuracy": metrics["accuracy"], "mae": metrics["mae"],
            "stability": metrics["stability"],
            "iterations": [{"k": 0, **{k: v for k, v in metrics.items() if k != "details"}}],
        }

    opt_llm = LLMClient(model=optimizer_model, temperature=1, logger=logger)
    optimizer = TextGradOptimizer(opt_llm)

    if separated:
        initial = (MEDICAL_EXTRACTOR_PROMPT, IMPAIRMENT_RATER_PROMPT, FINAL_AGGREGATOR_PROMPT)
    else:
        initial = (NAIVE_MEDICAL_EXTRACTOR_PROMPT, NAIVE_IMPAIRMENT_RATER_PROMPT,
                   NAIVE_FINAL_AGGREGATOR_PROMPT)
    current = {
        "medical_extractor": initial[0],
        "impairment_rater": initial[1],
        "final_aggregator": initial[2],
    }
    iteration_results = []

    for k in range(OPT_ITERATIONS):
        agents = _apply_retries(make_insurance_agents(
            extractor_prompt=current["medical_extractor"],
            rater_prompt=current["impairment_rater"],
            aggregator_prompt=current["final_aggregator"],
            separated=separated,
        ))
        batch = rng.sample(train_data, min(BATCH_SIZE, len(train_data)))

        graph = ComputationGraph()
        batch_correct = 0
        batch_stable = 0
        per_case: list[tuple[dict, str | None]] = []
        for applicant in batch:
            trace, predicted = run_insurance_episode(agents, applicant, llm)
            graph.add_from_trace(trace, current)
            if trace.completed_cleanly:
                batch_stable += 1
                if predicted == applicant["ground_truth"]:
                    batch_correct += 1
            per_case.append((applicant, predicted))

        batch_acc = batch_correct / len(batch)
        loss = 1.0 - batch_acc

        if feedback_mode == "rich":
            lines = [
                f"Batch accuracy: {batch_acc:.2f} ({batch_correct}/{len(batch)}), "
                f"stable: {batch_stable}/{len(batch)}",
                "Per-applicant outcomes:",
            ]
            for applicant, predicted in per_case:
                gt = applicant["ground_truth"]
                hit = "OK " if predicted == gt else "MISS"
                imp_summary = ", ".join(
                    f"{c['name']}({c['severity']})={c['rating']}" for c in applicant["conditions"]
                )
                lines.append(
                    f"  {hit}  age={applicant['age']}, {imp_summary}  "
                    f"=> predicted={predicted}, ground_truth={gt}"
                )
            feedback = "\n".join(lines)
        else:
            feedback = (
                f"Batch accuracy: {batch_acc:.2f} ({batch_correct}/{len(batch)}), "
                f"stable: {batch_stable}/{len(batch)}"
            )

        for name, agent in agents.items():
            new_prompt = optimizer.optimize(agent, graph, loss, feedback=feedback,
                                            separated=separated)
            logger.log_optimization_step(k, name, current[name], new_prompt, loss)
            current[name] = new_prompt

        eval_agents = _apply_retries(make_insurance_agents(
            extractor_prompt=current["medical_extractor"],
            rater_prompt=current["impairment_rater"],
            aggregator_prompt=current["final_aggregator"],
            separated=separated,
        ))
        test_m = evaluate_insurance(eval_agents, test_data, llm)
        val_m = evaluate_insurance(eval_agents, val_data, llm)
        row = {k2: v for k2, v in test_m.items() if k2 != "details"}
        row["val_accuracy"] = val_m["accuracy"]
        logger.log_metrics(k, {"method": method, "seed": seed, "iteration": k, **row})
        iteration_results.append({"k": k, **row, "mae": test_m["mae"], "stability": test_m["stability"]})
        print(f"    Iter {k}: acc={test_m['accuracy']:.3f} val={val_m['accuracy']:.3f}", flush=True)

    from experiments.common.splits import pick_best_iteration
    best = pick_best_iteration(iteration_results, "val_accuracy")
    return {
        "accuracy": best.get("accuracy", 0),
        "mae": best.get("mae", 0),
        "stability": best.get("stability", 0),
        "best_iter": best.get("k", 0),
        "iterations": iteration_results,
    }


def main():
    os.makedirs("results/insurance", exist_ok=True)
    all_results: dict = {}
    for method in ("fixed", "naive", "ours"):
        print(f"\n{'='*60}\nInsurance / {method}\n{'='*60}", flush=True)
        seed_results = []
        for seed in SEEDS:
            print(f"  Seed {seed}:", flush=True)
            logger = ExperimentLogger("insurance", f"{method}_s{seed}")
            r = run_insurance_experiment(method, seed, logger, feedback_mode="rich")
            seed_results.append(r)
            print(f"  -> acc={r['accuracy']:.3f}, mae={r['mae']:.3f}, "
                  f"stab={r['stability']:.3f}", flush=True)
        all_results[method] = {
            "mean_accuracy": round(sum(r["accuracy"] for r in seed_results) / len(seed_results), 4),
            "mean_mae": round(sum(r["mae"] for r in seed_results) / len(seed_results), 4),
            "mean_stability": round(sum(r["stability"] for r in seed_results) / len(seed_results), 4),
            "per_seed": seed_results,
        }
        print(f"  AVERAGE: "
              f"acc={all_results[method]['mean_accuracy']:.3f}  "
              f"mae={all_results[method]['mean_mae']:.3f}  "
              f"stab={all_results[method]['mean_stability']:.3f}")

    with open("results/insurance/results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\nINSURANCE (medical) SUMMARY")
    print("=" * 70)
    print(f"{'Method':<10} {'Accuracy':>10} {'MAE':>8} {'Stability':>10}")
    print("-" * 70)
    for name, r in all_results.items():
        print(f"{name:<10} {r['mean_accuracy']:>10.3f} {r['mean_mae']:>8.3f} {r['mean_stability']:>10.3f}")
    return all_results


if __name__ == "__main__":
    main()
