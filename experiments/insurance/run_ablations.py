"""Decomposition ablations for the insurance experiment, paralleling the
review ablations. Outputs to ``results/insurance/ablations.json``.

================ ============= ============ =============
Variant          Schema scaff  Parse retry  Rich feedback
================ ============= ============ =============
naive            no            no           scalar
schema_only      yes           no           scalar
schema_retry     yes           yes          scalar  (= existing "ours")
ours_full        yes           yes          rich
================ ============= ============ =============
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cdsep.logging_utils import ExperimentLogger
from experiments.insurance import run as insurance_run
from experiments.insurance.run import run_insurance_experiment


# Reduce iterations for ablations - we need the *qualitative* ranking of
# variants, not their best achievable scores (which are reported in tab:summary
# from the main run).
insurance_run.OPT_ITERATIONS = 4
SEEDS = insurance_run.SEEDS


VARIANTS = [
    ("naive",        dict(method="naive",                          feedback_mode="scalar")),
    ("schema_only",  dict(method="ours",  max_parse_retries=0,     feedback_mode="scalar")),
    ("schema_retry", dict(method="ours",                           feedback_mode="scalar")),
    ("ours_full",    dict(method="ours",                           feedback_mode="rich")),
]


def main() -> None:
    os.makedirs("results/insurance", exist_ok=True)
    out: dict[str, dict] = {}

    for variant_name, kwargs in VARIANTS:
        print(f"\n{'=' * 60}")
        print(f"Insurance ablation: {variant_name}")
        print(f"{'=' * 60}", flush=True)

        seed_results = []
        for seed in SEEDS:
            print(f"  Seed {seed}:", flush=True)
            logger = ExperimentLogger("insurance", f"abl_{variant_name}_s{seed}")
            result = run_insurance_experiment(seed=seed, logger=logger, **kwargs)
            seed_results.append(result)
            print(f"  -> acc={result.get('accuracy', 0):.3f} "
                  f"mae={result.get('mae', 0):.3f} "
                  f"stab={result.get('stability', 0):.3f}", flush=True)

        avg_keys = ["accuracy", "mae", "stability"]
        avgs = {f"mean_{k}": round(sum(r.get(k, 0) for r in seed_results) / len(seed_results), 4)
                for k in avg_keys}
        out[variant_name] = {**avgs, "per_seed": seed_results}
        print(f"  AVG: acc={avgs['mean_accuracy']:.3f} mae={avgs['mean_mae']:.3f} "
              f"stab={avgs['mean_stability']:.3f}")

    with open("results/insurance/ablations.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n\nABLATION SUMMARY")
    print("=" * 70)
    print(f"{'Variant':<14} {'Acc':>10} {'MAE':>10} {'Stability':>10}")
    print("-" * 70)
    for name, d in out.items():
        print(f"{name:<14} {d['mean_accuracy']:>10.3f} {d['mean_mae']:>10.3f} "
              f"{d['mean_stability']:>10.3f}")


if __name__ == "__main__":
    main()
