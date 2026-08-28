"""Decomposition ablations for the review experiment.

We disentangle ``ours`` along three orthogonal axes:

================ ============= ============ =============
Variant          Schema scaff  Parse retry  Rich feedback
================ ============= ============ =============
naive            no            no           scalar
schema_only      yes           no           scalar
schema_retry     yes           yes          scalar
ours_full        yes           yes          rich
================ ============= ============ =============

Run with the same seeds, splits, and MARG eval as ``run.py``. Outputs to
``results/review/ablations.json``.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cdsep.logging_utils import ExperimentLogger
from experiments.review import run as review_run
from experiments.review.run import run_review_experiment


# Reduce iterations for ablations to keep cost bounded; the main run already
# reports the headline numbers in tab:review.
review_run.OPT_ITERATIONS = 2
SEEDS = review_run.SEEDS


VARIANTS = [
    # name, kwargs override (separated, parse_retries, feedback_mode)
    ("naive",        dict(method="naive", separated_override=False, max_parse_retries=0,    feedback_mode="scalar")),
    ("schema_only",  dict(method="ours",  separated_override=True,  max_parse_retries=0,    feedback_mode="scalar")),
    ("schema_retry", dict(method="ours",  separated_override=True,  max_parse_retries=2,    feedback_mode="scalar")),
    ("ours_full",    dict(method="ours",  separated_override=True,  max_parse_retries=2,    feedback_mode="rich")),
]


def main() -> None:
    os.makedirs("results/review", exist_ok=True)
    out: dict[str, dict] = {}

    for variant_name, kwargs in VARIANTS:
        print(f"\n{'=' * 60}")
        print(f"Review ablation: {variant_name}")
        print(f"{'=' * 60}", flush=True)

        seed_results = []
        for seed in SEEDS:
            print(f"  Seed {seed}:", flush=True)
            logger = ExperimentLogger("review", f"abl_{variant_name}_s{seed}")
            result = run_review_experiment(seed=seed, logger=logger, **kwargs)
            seed_results.append(result)
            print(f"  -> R={result.get('recall', 0):.3f} "
                  f"P={result.get('precision', 0):.3f} "
                  f"J={result.get('jaccard', 0):.3f} "
                  f"stab={result.get('stability', 0):.3f}", flush=True)

        avg_keys = ["recall", "precision", "jaccard", "n_comments", "stability"]
        avgs = {f"mean_{k}": round(sum(r.get(k, 0) for r in seed_results) / len(seed_results), 4)
                for k in avg_keys}
        out[variant_name] = {**avgs, "per_seed": seed_results}
        print(f"  AVG: R={avgs['mean_recall']:.3f} P={avgs['mean_precision']:.3f} "
              f"J={avgs['mean_jaccard']:.3f} stab={avgs['mean_stability']:.3f}")

    with open("results/review/ablations.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n\nABLATION SUMMARY")
    print("=" * 70)
    print(f"{'Variant':<14} {'Recall':>10} {'Precision':>10} {'Jaccard':>10} {'Stability':>10}")
    print("-" * 70)
    for name, d in out.items():
        print(f"{name:<14} {d['mean_recall']:>10.3f} {d['mean_precision']:>10.3f} "
              f"{d['mean_jaccard']:>10.3f} {d['mean_stability']:>10.3f}")


if __name__ == "__main__":
    main()
