"""Reconstruct review results.json from per-run JSONL logs."""

from __future__ import annotations

import json
import os
import sys

LOG_DIR = "logs/review"
OUT_PATH = "results/review/results.json"


def parse_log(path: str) -> dict:
    """Walk a JSONL log and gather metrics events into iterations."""
    iterations = []
    last_metrics = None
    with open(path) as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("event") == "metrics":
                payload = {k: v for k, v in e.items()
                           if k in ("recall", "precision", "jaccard", "n_comments", "stability")}
                k_iter = e.get("iteration", 0) if "iteration" in e else 0
                payload["k"] = k_iter
                # Use accuracy as alias for jaccard for plotter compat
                payload["accuracy"] = e.get("jaccard", 0)
                iterations.append(payload)
                last_metrics = payload
    return iterations, last_metrics


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    methods = ["fixed", "naive", "ours"]
    seeds = [42, 123, 456]
    out = {}

    for method in methods:
        per_seed = []
        for seed in seeds:
            path = os.path.join(LOG_DIR, f"{method}_s{seed}.jsonl")
            if not os.path.exists(path):
                print(f"  Missing: {path}", file=sys.stderr)
                continue
            iters, last = parse_log(path)
            if not iters:
                print(f"  No metrics in {path}", file=sys.stderr)
                continue
            # For "fixed" there's only one metrics event (iteration 0)
            final = last if last else {}
            per_seed.append({
                **{k: v for k, v in final.items() if k != "k"},
                "iterations": iters,
            })

        if not per_seed:
            continue

        keys = ["recall", "precision", "jaccard", "n_comments", "stability"]
        avgs = {}
        for k in keys:
            vals = [r.get(k, 0) for r in per_seed if k in r]
            avgs[f"mean_{k}"] = round(sum(vals) / len(vals), 4) if vals else 0
        out[method] = {**avgs, "per_seed": per_seed}

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {OUT_PATH}")
    print()
    print(f"{'Method':<10} {'Recall':>10} {'Precision':>10} {'Jaccard':>10} {'#C':>6} {'Stab':>10}")
    print("-" * 65)
    for method, data in out.items():
        print(f"{method:<10} {data['mean_recall']:>10.3f} {data['mean_precision']:>10.3f} "
              f"{data['mean_jaccard']:>10.3f} {data.get('mean_n_comments',0):>6.1f} "
              f"{data['mean_stability']:>10.3f}")


if __name__ == "__main__":
    main()
