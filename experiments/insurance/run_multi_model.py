"""Multi-model insurance run: re-runs Fixed/Naive/Ours with Claude.

Outputs to ``results/insurance/multi_model.json``.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cdsep.logging_utils import ExperimentLogger
from experiments.insurance import run as insurance_run
from experiments.insurance.run import run_insurance_experiment


# Cap iterations to keep cost bounded; main paper table reports the
# full-iteration OpenAI numbers separately.
insurance_run.OPT_ITERATIONS = 3
SEEDS = [42, 123]


CONFIGS = [
    ("openai",     "gpt-5.4-nano",     "gpt-5.4-mini"),
    ("anthropic",  "claude-haiku-4-5", "claude-sonnet-4-5"),
]


def main() -> None:
    os.makedirs("results/insurance", exist_ok=True)
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
                logger = ExperimentLogger("insurance", f"mm_{fam_name}_{method}_s{seed}")
                try:
                    r = run_insurance_experiment(
                        method=method,
                        seed=seed,
                        logger=logger,
                        agent_model=agent_model,
                        optimizer_model=optimizer_model,
                    )
                except Exception as e:
                    print(f"  -> CRASHED: {type(e).__name__}: {e}", flush=True)
                    r = {"accuracy": 0, "mae": 1.0, "stability": 0, "iterations": []}
                seed_results.append(r)
                print(f"  -> acc={r.get('accuracy', 0):.3f} "
                      f"mae={r.get('mae', 0):.3f} "
                      f"stab={r.get('stability', 0):.3f}", flush=True)

            keys = ["accuracy", "mae", "stability"]
            avgs = {f"mean_{k}": round(
                sum(r.get(k, 0) for r in seed_results) / max(1, len(seed_results)), 4
            ) for k in keys}
            out[fam_name][method] = {**avgs, "per_seed": seed_results}
            print(f"  AVG: acc={avgs['mean_accuracy']:.3f} mae={avgs['mean_mae']:.3f} "
                  f"stab={avgs['mean_stability']:.3f}")

        with open("results/insurance/multi_model.json", "w") as f:
            json.dump(out, f, indent=2, default=str)

    print("\n\nMULTI-MODEL INSURANCE SUMMARY")
    print("=" * 80)
    print(f"{'Family':<12} {'Method':<8} {'Acc':>10} {'MAE':>10} {'Stab':>10}")
    print("-" * 80)
    for fam, by_method in out.items():
        for m, d in by_method.items():
            print(f"{fam:<12} {m:<8} {d['mean_accuracy']:>10.3f} "
                  f"{d['mean_mae']:>10.3f} {d['mean_stability']:>10.3f}")
        print()


if __name__ == "__main__":
    main()
