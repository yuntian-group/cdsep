"""Multi-model robustness study for the review experiment.

Re-runs Fixed / Naive / Ours on the same MARG benchmark with three different
LLM families:

* OpenAI:   gpt-5.4-nano agents,        gpt-5.4-mini optimizer
* Anthropic: claude-haiku-4-5 agents,   claude-sonnet-4-5 optimizer
* Google:   gemini-2.5-flash agents,    gemini-2.5-pro optimizer

Outputs ``results/review/multi_model.json`` with the per-(model, method) metrics.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cdsep.logging_utils import ExperimentLogger
from experiments.review import run as review_run
from experiments.review.run import run_review_experiment


# We compare three model families. OpenAI numbers are also reported in the
# main paper table; we re-run them here only for "fixed" so the multi-model
# table is self-contained. The optimization loop is shortened to 2 iterations
# per (family, method, seed) for the multi-model panel since the goal is to
# confirm that ours' relative gain holds across families, not to reach the
# absolute best score for each family.
review_run.OPT_ITERATIONS = 2

# Use just 1 seed per family for cost reasons; this is acceptable because the
# PER-FAMILY result is the new contribution; the cross-seed variance is
# already characterized for OpenAI in tab:review.
SEEDS = [42]


CONFIGS = [
    ("openai",    "gpt-5.4-nano",     "gpt-5.4-mini"),
    ("anthropic", "claude-haiku-4-5", "claude-sonnet-4-5"),
    ("google",    "gemini-2.5-flash", "gemini-2.5-pro"),
]


def main() -> None:
    os.makedirs("results/review", exist_ok=True)
    out: dict[str, dict[str, dict]] = {}

    for fam_name, agent_model, optimizer_model in CONFIGS:
        out[fam_name] = {}
        for method in ("fixed", "naive", "ours"):
            print(f"\n{'=' * 60}")
            print(f"{fam_name.upper():<10} / {method}")
            print(f"  agents = {agent_model}, optimizer = {optimizer_model}")
            print(f"{'=' * 60}", flush=True)

            seed_results = []
            for seed in SEEDS:
                print(f"  Seed {seed}:", flush=True)
                logger = ExperimentLogger("review", f"mm_{fam_name}_{method}_s{seed}")
                kwargs = dict(
                    method=method,
                    seed=seed,
                    logger=logger,
                    agent_model=agent_model,
                    optimizer_model=optimizer_model,
                )
                try:
                    r = run_review_experiment(**kwargs)
                except Exception as e:
                    print(f"  -> CRASHED: {type(e).__name__}: {e}", flush=True)
                    r = {"recall": 0, "precision": 0, "jaccard": 0, "stability": 0,
                         "n_comments": 0, "iterations": []}
                seed_results.append(r)
                print(f"  -> R={r.get('recall', 0):.3f} "
                      f"P={r.get('precision', 0):.3f} "
                      f"J={r.get('jaccard', 0):.3f} "
                      f"stab={r.get('stability', 0):.3f}", flush=True)

            keys = ["recall", "precision", "jaccard", "n_comments", "stability"]
            avgs = {f"mean_{k}": round(
                sum(r.get(k, 0) for r in seed_results) / max(1, len(seed_results)), 4
            ) for k in keys}
            out[fam_name][method] = {**avgs, "per_seed": seed_results}
            print(f"  AVG: R={avgs['mean_recall']:.3f} P={avgs['mean_precision']:.3f} "
                  f"J={avgs['mean_jaccard']:.3f} stab={avgs['mean_stability']:.3f}")

        # Persist incrementally so we don't lose work on a crash
        with open("results/review/multi_model.json", "w") as f:
            json.dump(out, f, indent=2, default=str)

    print("\n\nMULTI-MODEL SUMMARY")
    print("=" * 80)
    print(f"{'Family':<12} {'Method':<8} {'Recall':>10} {'Precision':>10} {'Jaccard':>10} {'Stab':>10}")
    print("-" * 80)
    for fam, by_method in out.items():
        for m, d in by_method.items():
            print(f"{fam:<12} {m:<8} {d['mean_recall']:>10.3f} "
                  f"{d['mean_precision']:>10.3f} {d['mean_jaccard']:>10.3f} "
                  f"{d['mean_stability']:>10.3f}")
        print()


if __name__ == "__main__":
    main()
