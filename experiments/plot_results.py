"""Generate result plots and tables from experiment outputs."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(experiment: str) -> dict:
    path = f"results/{experiment}/results.json"
    with open(path) as f:
        return json.load(f)


def plot_synthetic_optimization_curves(results: dict, output_dir: str = "results/synthetic"):
    """Plot accuracy vs optimization iteration for each function."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    func_names = ["Max", "Min", "ModSum10", "Multiply"]

    for ax, func_name in zip(axes, func_names):
        for method, style in [("naive", {"color": "C1", "ls": "--", "label": "Naïve TextGrad"}),
                               ("ours", {"color": "C2", "ls": "-", "label": "Ours (separated)"})]:
            if func_name not in results or method not in results[func_name]:
                continue
            data = results[func_name][method]
            all_iters = []
            for seed_result in data.get("per_seed", []):
                iters = seed_result.get("iterations", [])
                if iters and "k" in iters[0]:
                    accs = [it["accuracy"] for it in iters]
                    all_iters.append(accs)
            if all_iters:
                max_len = max(len(a) for a in all_iters)
                padded = [a + [a[-1]] * (max_len - len(a)) for a in all_iters]
                mean = np.mean(padded, axis=0)
                std = np.std(padded, axis=0)
                x = range(len(mean))
                ax.plot(x, mean, **{k: v for k, v in style.items() if k != "label"}, label=style["label"])
                ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=style.get("color"))

        if func_name in results and "fixed" in results[func_name]:
            fixed_acc = results[func_name]["fixed"]["mean_accuracy"]
            ax.axhline(y=fixed_acc, color="C0", ls=":", label="Fixed prompts")

        ax.set_title(func_name)
        ax.set_xlabel("Optimization Iteration")
        if ax == axes[0]:
            ax.set_ylabel("Test Accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optimization_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_dir}/optimization_curves.png")


def plot_insurance_optimization_curves(results: dict, output_dir: str = "results/insurance"):
    """Plot accuracy and stability curves for insurance experiment."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for method, style in [("naive", {"color": "C1", "ls": "--", "label": "Naïve TextGrad"}),
                           ("ours", {"color": "C2", "ls": "-", "label": "Ours (separated)"})]:
        if method not in results:
            continue
        data = results[method]
        all_accs = []
        all_stabs = []
        for seed_result in data.get("per_seed", []):
            iters = seed_result.get("iterations", [])
            if iters and "k" in iters[0]:
                all_accs.append([it["accuracy"] for it in iters])
                all_stabs.append([it["stability"] for it in iters])
        if all_accs:
            max_len = max(len(a) for a in all_accs)
            padded_a = [a + [a[-1]] * (max_len - len(a)) for a in all_accs]
            padded_s = [s + [s[-1]] * (max_len - len(s)) for s in all_stabs]
            mean_a = np.mean(padded_a, axis=0)
            mean_s = np.mean(padded_s, axis=0)
            x = range(len(mean_a))
            ax1.plot(x, mean_a, color=style["color"], ls=style["ls"], label=style["label"])
            ax2.plot(x, mean_s, color=style["color"], ls=style["ls"], label=style["label"])

    if "fixed" in results:
        ax1.axhline(y=results["fixed"]["mean_accuracy"], color="C0", ls=":", label="Fixed")
        ax2.axhline(y=results["fixed"]["mean_stability"], color="C0", ls=":", label="Fixed")

    ax1.set_title("Insurance Rating Accuracy")
    ax1.set_xlabel("Optimization Iteration")
    ax1.set_ylabel("Accuracy")
    ax1.legend(fontsize=8)

    ax2.set_title("Insurance Stability")
    ax2.set_xlabel("Optimization Iteration")
    ax2.set_ylabel("Stability")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optimization_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_dir}/optimization_curves.png")


def main():
    for exp in ["synthetic", "insurance", "review"]:
        path = f"results/{exp}/results.json"
        if os.path.exists(path):
            print(f"\n=== {exp.upper()} ===")
            results = load_results(exp)
            if exp == "synthetic":
                plot_synthetic_optimization_curves(results)
            elif exp == "insurance":
                plot_insurance_optimization_curves(results)
            print(json.dumps({k: {k2: v2 for k2, v2 in v.items() if k2 != "per_seed"} if isinstance(v, dict) else v for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
