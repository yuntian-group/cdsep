"""Run collaborative review generation experiments on the MARG benchmark.

Optimization protocol (for naive / ours):
* Split papers into train / val / test.
* Each iteration:
    - sample a batch of train papers, run pipeline, gather (predicted, reference)
      atomic comments and per-paper metrics
    - feed the optimizer a *rich* feedback string that names specific reference
      comments we missed and predicted comments that were not aligned, plus the
      scalar Jaccard
    - update leader and worker prompts independently
    - evaluate on the val split
* Report the test metrics from the iteration with the best val Jaccard
  ("best-iter selection"). Always include iter 0 as the no-op baseline.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.review.agents import (
    LEADER_PROMPT,
    NAIVE_LEADER_PROMPT,
    NAIVE_WORKER_PROMPT,
    WORKER_PROMPT,
    LeaderControl,
    WorkerControl,
    make_review_agents,
)
from experiments.review.configs import (
    BATCH_SIZE,
    MODEL,
    N_PAPERS,
    N_TEST,
    N_TRAIN,
    N_VAL,
    OPT_ITERATIONS,
    OPTIMIZER_MODEL,
    SEEDS,
)
from experiments.review.marg_data import load_papers as load_marg_papers
from experiments.review.marg_metrics import (
    alignment_metrics,
    compute_metrics_marg as compute_metrics,
    extract_reference_comments,
)
from cdsep.agent import Agent
from cdsep.episode import EpisodeTrace, run_episode
from cdsep.graph import ComputationGraph
from cdsep.llm import LLMClient
from cdsep.logging_utils import ExperimentLogger
from cdsep.optimizer import TextGradOptimizer


def split_three_ways(papers: list[dict], n_train: int, n_val: int) -> tuple[list[dict], list[dict], list[dict]]:
    return papers[:n_train], papers[n_train:n_train + n_val], papers[n_train + n_val:]


def route_review(control) -> str:
    if isinstance(control, LeaderControl):
        if control.stop:
            return "terminate"
        return control.target_agent
    if isinstance(control, WorkerControl):
        return "leader"
    return "terminate"


def format_paper_input(paper: dict) -> str:
    lines = [f"Title: {paper['title']}", f"Abstract: {paper['abstract']}", ""]
    for sec_name, sec_text in paper["sections"].items():
        lines.append(f"## {sec_name.title()}")
        lines.append(sec_text)
        lines.append("")
    return "\n".join(lines)


def extract_comments_from_message(message: str) -> list[str]:
    """Extract atomic comments from a numbered list in the message."""
    comments = []
    for line in message.split("\n"):
        line = line.strip()
        match = re.match(r'^[\d]+[.)]\s*(.+)', line)
        if match:
            comments.append(match.group(1).strip())
        elif line.startswith("- ") and len(line) > 10:
            comments.append(line[2:].strip())
    if not comments and len(message) > 20:
        for sent in message.split(". "):
            sent = sent.strip()
            if len(sent) > 15:
                comments.append(sent)
    return comments


def run_review_episode(agents: dict[str, Agent], paper: dict, llm: LLMClient) -> tuple[EpisodeTrace, list[str]]:
    """Run one review episode, return (trace, extracted_comments)."""
    trace = run_episode(
        entry_agent=agents["leader"],
        agents=agents,
        route_fn=route_review,
        task_input=format_paper_input(paper),
        llm=llm,
        max_steps=12,
    )

    comments = []
    if trace.outcome == "completed" and trace.steps:
        final_msg = trace.steps[-1].message
        comments = extract_comments_from_message(final_msg)
    if not comments:
        for step in trace.steps:
            if step.agent_name.startswith("worker"):
                comments.extend(extract_comments_from_message(step.message))

    return trace, comments


def evaluate_review(agents: dict[str, Agent], test_papers: list[dict], llm: LLMClient, eval_llm: LLMClient):
    """Evaluate review pipeline on test papers (parallelised across papers)."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    total_stable = 0
    total_completed = 0

    def _run_one(paper):
        trace, system_comments = run_review_episode(agents, paper, llm)
        m = compute_metrics(system_comments, paper["human_reviews"], eval_llm)
        m["stable"] = trace.is_stable
        m["completed"] = trace.completed_cleanly
        return m

    workers = max(1, min(int(os.environ.get("REVIEW_EVAL_WORKERS", "8")), len(test_papers)))
    if workers <= 1:
        all_metrics = [_run_one(p) for p in test_papers]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            all_metrics = list(ex.map(_run_one, test_papers))
    for m in all_metrics:
        if m["stable"]:
            total_stable += 1
        if m["completed"]:
            total_completed += 1

    n = len(test_papers)
    avg = {
        "recall": sum(m["recall"] for m in all_metrics) / n if n else 0,
        "precision": sum(m["precision"] for m in all_metrics) / n if n else 0,
        "jaccard": sum(m["jaccard"] for m in all_metrics) / n if n else 0,
        "n_comments": sum(m["n_comments"] for m in all_metrics) / n if n else 0,
        "stability": total_stable / n if n else 0,
        "completion_rate": total_completed / n if n else 0,
    }
    return avg, all_metrics


def _truncate(s: str, n: int = 220) -> str:
    s = s.strip().replace("\n", " ")
    return s if len(s) <= n else s[:n] + "..."


def build_optimizer_feedback(
    batch_papers: list[dict],
    batch_predicted: list[list[str]],
    batch_reference: list[list[str]],
    batch_align: list[list[list[bool]]],
    batch_metrics: list[dict],
) -> str:
    """Construct a rich textual gradient that names *specific* reference
    comments we missed and predicted comments that were not aligned.

    This gives the optimizer concrete material to compare against, instead of
    a single scalar Jaccard.
    """
    lines = []
    avg_jaccard = sum(m["jaccard"] for m in batch_metrics) / len(batch_metrics) if batch_metrics else 0
    lines.append(f"Batch summary: avg Jaccard {avg_jaccard:.3f} over {len(batch_metrics)} papers.")
    lines.append(
        "Below, for each paper, are the reference (human) comments your system "
        "MISSED, and the system comments that did NOT align to anything. "
        "Use these to make prompt edits that help future drafts cover the "
        "missed topics and avoid generating the unaligned ones."
    )
    for i, (paper, pred, ref, align, m) in enumerate(zip(
        batch_papers, batch_predicted, batch_reference, batch_align, batch_metrics
    )):
        title = paper.get("title", f"paper {i}")[:80]
        lines.append("")
        lines.append(f"### Paper {i + 1}: {title}")
        lines.append(f"Recall {m['recall']:.2f}, Precision {m['precision']:.2f}, "
                     f"Jaccard {m['jaccard']:.2f}, "
                     f"{len(pred)} predicted, {len(ref)} reference.")

        # Reference comments NOT aligned (= missed by us)
        if align and pred and ref:
            ref_aligned = {j for row in align for j, v in enumerate(row) if v}
            missed = [r for j, r in enumerate(ref) if j not in ref_aligned]
            if missed:
                lines.append("MISSED REFERENCE COMMENTS (try to cover these):")
                for r in missed[:5]:
                    lines.append(f"  - {_truncate(r)}")
            # Predicted not aligned
            pred_aligned = {j for j, row in enumerate(align) if any(row)}
            extras = [p for j, p in enumerate(pred) if j not in pred_aligned]
            if extras:
                lines.append("UNALIGNED PREDICTED COMMENTS (these did not match any "
                             "reference; they may still be useful but did not score):")
                for p in extras[:5]:
                    lines.append(f"  - {_truncate(p)}")
        elif not pred:
            lines.append("NO PREDICTED COMMENTS WERE PRODUCED for this paper.")
        elif not ref:
            lines.append("(reference comments empty for this paper)")
    return "\n".join(lines)


def run_review_experiment(
    method: str,
    seed: int,
    logger: ExperimentLogger,
    *,
    agent_model: str | None = None,
    optimizer_model: str | None = None,
    separated_override: bool | None = None,
    feedback_mode: str = "rich",        # "rich" | "scalar"
    max_parse_retries: int | None = None,
):
    """Run one review experiment on the MARG/ARIES benchmark.

    Parameters beyond ``method`` / ``seed`` / ``logger`` enable ablations
    and multi-model variants while keeping the main code path intact:

    - ``agent_model``         : override per-agent LLM (default = configs.MODEL)
    - ``optimizer_model``     : override the optimizer LLM (default = configs.OPTIMIZER_MODEL)
    - ``separated_override``  : force schema-scaffolding on/off (else inferred from method)
    - ``feedback_mode``       : ``"rich"`` (full per-paper feedback) or
                                ``"scalar"`` (just the batch Jaccard)
    - ``max_parse_retries``   : override the agent parse-retry budget
    """
    papers = load_marg_papers(seed=seed, limit=N_PAPERS)
    train_papers, val_papers, test_papers = split_three_ways(papers, N_TRAIN, N_VAL)
    rng = random.Random(seed)

    agent_model = agent_model or MODEL
    optimizer_model = optimizer_model or OPTIMIZER_MODEL
    llm = LLMClient(model=agent_model, temperature=0, logger=logger)
    eval_llm = LLMClient(model=agent_model, temperature=0, logger=logger)

    separated = (method == "ours") if separated_override is None else separated_override

    def _apply_retries(agents):
        if max_parse_retries is not None:
            for a in agents.values():
                a.max_parse_retries = max_parse_retries
        return agents

    if method == "fixed":
        agents = _apply_retries(make_review_agents(separated=True))
        avg, _ = evaluate_review(agents, test_papers, llm, eval_llm)
        logger.log_metrics(0, {"method": method, "seed": seed, **avg})
        return {**avg, "iterations": [{"k": 0, **avg}]}

    opt_llm = LLMClient(model=optimizer_model, temperature=1, logger=logger)
    optimizer = TextGradOptimizer(opt_llm)

    if separated:
        current_prompts = {"leader": LEADER_PROMPT, "worker": WORKER_PROMPT}
    else:
        current_prompts = {"leader": NAIVE_LEADER_PROMPT, "worker": NAIVE_WORKER_PROMPT}

    # ---- iter 0: evaluate the initial prompts on val and test ----
    iteration_results = []
    best_prompts = dict(current_prompts)
    best_val_jaccard = -1.0

    def _build_eval_agents(prompts):
        return _apply_retries(make_review_agents(
            leader_prompt=prompts["leader"],
            worker_prompt=prompts["worker"],
            separated=separated,
        ))

    val0_agents = _build_eval_agents(current_prompts)
    val0, _ = evaluate_review(val0_agents, val_papers, llm, eval_llm)
    test0, _ = evaluate_review(val0_agents, test_papers, llm, eval_llm)
    best_val_jaccard = val0["jaccard"]
    iteration_results.append({"k": 0, **test0, "val_jaccard": val0["jaccard"]})
    logger.log_metrics(0, {"method": method, "seed": seed, "iteration": 0,
                            "split": "test", **test0, "val_jaccard": val0["jaccard"]})
    print(f"    Iter 0 (init): val_J={val0['jaccard']:.3f} | test "
          f"R={test0['recall']:.3f} P={test0['precision']:.3f} J={test0['jaccard']:.3f}",
          flush=True)

    # ---- optimization loop ----
    for k in range(1, OPT_ITERATIONS + 1):
        agents = _build_eval_agents(current_prompts)

        batch = rng.sample(train_papers, min(BATCH_SIZE, len(train_papers)))

        graph = ComputationGraph()
        batch_pred, batch_ref, batch_align, batch_metrics = [], [], [], []

        for paper in batch:
            trace, system_comments = run_review_episode(agents, paper, llm)
            prompt_map = {"leader": current_prompts["leader"]}
            for wn in ["worker_1", "worker_2", "worker_3"]:
                prompt_map[wn] = current_prompts["worker"]
            graph.add_from_trace(trace, prompt_map)

            ref = extract_reference_comments(paper["human_reviews"], eval_llm)
            res = alignment_metrics(system_comments, ref, eval_llm)
            batch_pred.append(system_comments[:res.n_predicted])
            batch_ref.append(ref[:res.n_reference])
            batch_align.append(res.alignment_matrix or [])
            batch_metrics.append({
                "recall": res.recall, "precision": res.precision, "jaccard": res.jaccard,
            })

        avg_jaccard = (sum(m["jaccard"] for m in batch_metrics) / len(batch_metrics)
                       if batch_metrics else 0)
        loss = 1.0 - avg_jaccard
        if feedback_mode == "rich":
            feedback = build_optimizer_feedback(
                batch, batch_pred, batch_ref, batch_align, batch_metrics
            )
        else:  # "scalar"
            feedback = f"Batch avg Jaccard: {avg_jaccard:.3f}"

        # Optimize each agent independently
        leader_agent = agents["leader"]
        new_leader = optimizer.optimize(leader_agent, graph, loss,
                                        feedback=feedback, separated=separated)
        logger.log_optimization_step(k, "leader", current_prompts["leader"],
                                     new_leader, loss)
        current_prompts["leader"] = new_leader

        worker_agent = agents["worker_1"]
        new_worker = optimizer.optimize(worker_agent, graph, loss,
                                        feedback=feedback, separated=separated)
        logger.log_optimization_step(k, "worker", current_prompts["worker"],
                                     new_worker, loss)
        current_prompts["worker"] = new_worker

        eval_agents = _build_eval_agents(current_prompts)
        val_avg, _ = evaluate_review(eval_agents, val_papers, llm, eval_llm)
        test_avg, _ = evaluate_review(eval_agents, test_papers, llm, eval_llm)
        if val_avg["jaccard"] > best_val_jaccard:
            best_val_jaccard = val_avg["jaccard"]
            best_prompts = dict(current_prompts)

        iteration_results.append({"k": k, **test_avg, "val_jaccard": val_avg["jaccard"]})
        logger.log_metrics(k, {"method": method, "seed": seed, "iteration": k,
                                "split": "test", **test_avg,
                                "val_jaccard": val_avg["jaccard"]})
        print(f"    Iter {k}: val_J={val_avg['jaccard']:.3f} | test "
              f"R={test_avg['recall']:.3f} P={test_avg['precision']:.3f} "
              f"J={test_avg['jaccard']:.3f} | batch_loss={loss:.3f}", flush=True)

    # ---- pick best-iter result for reporting ----
    best_iter = max(range(len(iteration_results)),
                    key=lambda i: iteration_results[i].get("val_jaccard", -1))
    best = iteration_results[best_iter]
    print(f"  -> picked iter {best['k']} (val_J={best.get('val_jaccard', 0):.3f}); "
          f"test R={best['recall']:.3f} P={best['precision']:.3f} "
          f"J={best['jaccard']:.3f}", flush=True)
    return {
        "recall": best["recall"],
        "precision": best["precision"],
        "jaccard": best["jaccard"],
        "n_comments": best["n_comments"],
        "stability": best["stability"],
        "best_iter": best["k"],
        "iterations": iteration_results,
        "best_prompts": best_prompts,
    }


def main():
    os.makedirs("results/review", exist_ok=True)
    all_results = {}

    for method in ["fixed", "naive", "ours"]:
        print(f"\n{'='*60}")
        print(f"Review / {method}")
        print(f"{'='*60}", flush=True)

        seed_results = []
        for seed in SEEDS:
            print(f"  Seed {seed}:", flush=True)
            logger = ExperimentLogger("review", f"{method}_s{seed}")
            result = run_review_experiment(method, seed, logger)
            seed_results.append(result)
            print(f"  -> R={result.get('recall',0):.3f}, "
                  f"P={result.get('precision',0):.3f}, "
                  f"J={result.get('jaccard',0):.3f}", flush=True)

        avg_keys = ["recall", "precision", "jaccard", "n_comments", "stability"]
        avgs = {}
        for key in avg_keys:
            vals = [r.get(key, 0) for r in seed_results]
            avgs[f"mean_{key}"] = round(sum(vals) / len(vals), 4) if vals else 0

        all_results[method] = {**avgs, "per_seed": seed_results}
        print(f"  AVERAGE: R={avgs['mean_recall']:.3f}, "
              f"P={avgs['mean_precision']:.3f}, J={avgs['mean_jaccard']:.3f}")

    with open("results/review/results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n\nREVIEW SUMMARY")
    print("=" * 70)
    print(f"{'Method':<10} {'Recall':>10} {'Precision':>10} {'Jaccard':>10} "
          f"{'#Comments':>10} {'Stability':>10}")
    print("-" * 70)
    for method, r in all_results.items():
        print(f"{method:<10} {r['mean_recall']:>10.3f} {r['mean_precision']:>10.3f} "
              f"{r['mean_jaccard']:>10.3f} {r.get('mean_n_comments',0):>10.1f} "
              f"{r['mean_stability']:>10.3f}")

    return all_results


if __name__ == "__main__":
    main()
