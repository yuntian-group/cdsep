"""Generate publication-quality figures for the paper."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

COLORS = {
    "fixed": "#5B8DB8",
    "naive": "#E8853D",
    "ours":  "#4DAF4A",
}
LABELS = {
    "fixed": "Fixed prompts",
    "naive": "Naïve TextGrad",
    "ours":  "Ours (separated)",
}


def _gather_curves(data: dict, methods=("naive", "ours")) -> dict:
    """Extract per-iteration accuracy curves from results."""
    curves = {}
    for method in methods:
        all_iters = []
        for seed_result in data.get("per_seed", []):
            iters = seed_result.get("iterations", [])
            if iters and "k" in iters[0]:
                all_iters.append([it["accuracy"] for it in iters])
        if all_iters:
            max_len = max(len(a) for a in all_iters)
            padded = [a + [a[-1]] * (max_len - len(a)) for a in all_iters]
            curves[method] = {
                "mean": np.mean(padded, axis=0),
                "std": np.std(padded, axis=0),
            }
    return curves


def plot_synthetic_curves(results: dict, output_path: str):
    func_names = ["Max", "Min", "ModSum10", "Multiply"]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.8), sharey=True)

    for ax, func in zip(axes, func_names):
        fixed_acc = results[func]["fixed"]["mean_accuracy"]
        ax.axhline(y=fixed_acc, color=COLORS["fixed"], ls=":", lw=1.2,
                    label=LABELS["fixed"], zorder=1)

        for method in ["naive", "ours"]:
            curves = _gather_curves(results[func][method])
            if method in curves:
                m = curves[method]["mean"]
                s = curves[method]["std"]
                x = np.arange(len(m))
                ax.plot(x, m, color=COLORS[method], lw=1.5, label=LABELS[method], zorder=2)
                ax.fill_between(x, m - s, np.minimum(m + s, 1.05), alpha=0.15, color=COLORS[method])

        ax.set_title(func, fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.set_ylim(0.15, 1.07)
        ax.set_xlim(-0.3, 7.3)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.grid(axis="y", ls="--", alpha=0.3)

    axes[0].set_ylabel("Test Accuracy")
    axes[-1].legend(loc="lower right", framealpha=0.9, edgecolor="none")

    plt.tight_layout(w_pad=0.5)
    fig.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def plot_insurance_curves(results: dict, output_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.0, 2.2))

    fixed_acc = results["fixed"]["mean_accuracy"]
    fixed_mae = results["fixed"]["mean_mae"]

    for method in ["naive", "ours"]:
        data = results[method]
        all_accs, all_maes = [], []
        for sr in data.get("per_seed", []):
            iters = sr.get("iterations", [])
            if iters and "k" in iters[0]:
                all_accs.append([it["accuracy"] for it in iters])
                all_maes.append([it["mae"] for it in iters])
        if all_accs:
            max_len = max(len(a) for a in all_accs)
            pa = [a + [a[-1]] * (max_len - len(a)) for a in all_accs]
            pm = [a + [a[-1]] * (max_len - len(a)) for a in all_maes]
            ma, sa = np.mean(pa, axis=0), np.std(pa, axis=0)
            mm, sm = np.mean(pm, axis=0), np.std(pm, axis=0)
            x = np.arange(len(ma))
            ax1.plot(x, ma, color=COLORS[method], lw=1.5, label=LABELS[method])
            ax1.fill_between(x, ma - sa, np.minimum(ma + sa, 1.0), alpha=0.15, color=COLORS[method])
            ax2.plot(x, mm, color=COLORS[method], lw=1.5, label=LABELS[method])
            ax2.fill_between(x, np.maximum(mm - sm, 0), mm + sm, alpha=0.15, color=COLORS[method])

    ax1.axhline(y=fixed_acc, color=COLORS["fixed"], ls=":", lw=1.2, label=LABELS["fixed"])
    ax2.axhline(y=fixed_mae, color=COLORS["fixed"], ls=":", lw=1.2, label=LABELS["fixed"])

    ax1.set_title("Accuracy", fontweight="bold")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Accuracy")
    ax1.grid(axis="y", ls="--", alpha=0.3)

    ax2.set_title("MAE", fontweight="bold")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("MAE")
    ax2.grid(axis="y", ls="--", alpha=0.3)
    ax2.legend(loc="upper right", framealpha=0.9, edgecolor="none")

    plt.tight_layout(w_pad=1.0)
    fig.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def plot_ablation_bar(results: dict, output_path: str):
    methods = ["ours", "no_validation", "editable_control", "fixed"]
    labels = ["Full system\n(ours)", "No\nvalidation", "Editable\ncontrol", "Freeze\nprompts"]
    # The new (medical-impairment) insurance experiment doesn't include the
    # no_validation / editable_control ablations -- those were specific to the
    # earlier demographic version. Skip the figure gracefully if any key is
    # missing rather than crashing the whole figure pass.
    if not all(m in results for m in methods):
        print(f"Skipping {output_path}: ablation keys not present in results "
              f"(have {sorted(results.keys())})")
        return
    accs = [results[m]["mean_accuracy"] for m in methods]
    maes = [results[m]["mean_mae"] for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.0, 2.0))

    colors = ["#4DAF4A", "#E8853D", "#E8853D", "#5B8DB8"]
    x = np.arange(len(methods))
    ax1.bar(x, accs, color=colors, width=0.6, edgecolor="white", lw=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy by Variant", fontweight="bold")
    ax1.set_ylim(0, 0.75)
    ax1.grid(axis="y", ls="--", alpha=0.3)

    ax2.bar(x, maes, color=colors, width=0.6, edgecolor="white", lw=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("MAE")
    ax2.set_title("MAE by Variant", fontweight="bold")
    ax2.set_ylim(0, 0.75)
    ax2.grid(axis="y", ls="--", alpha=0.3)

    plt.tight_layout(w_pad=1.0)
    fig.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def plot_review_stability(results: dict, output_path: str):
    """Bar chart showing stability collapse for naive on review."""
    methods = ["fixed", "naive", "ours"]
    labels = ["Fixed", "Naïve\nTextGrad", "Ours\n(separated)"]
    stabs = [results[m]["mean_stability"] * 100 for m in methods]
    jaccs = [results[m]["mean_jaccard"] * 100 for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.0, 2.0))
    colors = [COLORS["fixed"], COLORS["naive"], COLORS["ours"]]
    x = np.arange(len(methods))

    bars1 = ax1.bar(x, stabs, color=colors, width=0.6, edgecolor="white", lw=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Stability (%)")
    ax1.set_title("Episode Stability", fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.grid(axis="y", ls="--", alpha=0.3)
    for b, v in zip(bars1, stabs):
        ax1.text(b.get_x() + b.get_width()/2, v + 2, f"{v:.0f}%",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")

    bars2 = ax2.bar(x, jaccs, color=colors, width=0.6, edgecolor="white", lw=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Jaccard (%)")
    ax2.set_title("Jaccard Score", fontweight="bold")
    ax2.set_ylim(0, max(jaccs) * 1.3 + 1)
    ax2.grid(axis="y", ls="--", alpha=0.3)
    for b, v in zip(bars2, jaccs):
        ax2.text(b.get_x() + b.get_width()/2, v + 0.1, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout(w_pad=1.0)
    fig.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def plot_bbh_curves(results: dict, output_path: str):
    """Same shape as synthetic curves but adapted for BBH tasks."""
    task_names = list(results.keys())
    n = len(task_names)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(1.8 * n + 0.4, 1.8), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, task in zip(axes, task_names):
        if task in results and "fixed" in results[task]:
            ax.axhline(y=results[task]["fixed"]["mean_accuracy"], color=COLORS["fixed"],
                        ls=":", lw=1.2, label=LABELS["fixed"])
        for method in ("naive", "ours"):
            curves = _gather_curves(results[task].get(method, {}))
            if method in curves:
                m = curves[method]["mean"]
                s = curves[method]["std"]
                x = np.arange(len(m))
                ax.plot(x, m, color=COLORS[method], lw=1.5, label=LABELS[method])
                ax.fill_between(x, np.maximum(m - s, 0),
                                np.minimum(m + s, 1.05), alpha=0.15, color=COLORS[method])
        # Pretty-print task name
        short = (task.replace("_three_objects", "")
                     .replace("logical_deduction", "LogDed")
                     .replace("tracking_shuffled_objects", "TrkObj")
                     .replace("causal_judgement", "CausJudg")
                     .replace("word_sorting", "WordSort"))
        ax.set_title(short, fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.set_ylim(0.0, 1.05)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.grid(axis="y", ls="--", alpha=0.3)

    axes[0].set_ylabel("Test Accuracy")
    axes[-1].legend(loc="lower right", framealpha=0.9, edgecolor="none", fontsize=7)

    plt.tight_layout(w_pad=0.5)
    fig.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def plot_multi_model(results: dict, output_path: str) -> None:
    """Group bar chart: families on x-axis, methods as bars within each group."""
    families = list(results.keys())
    methods = ["fixed", "naive", "ours"]

    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    x = np.arange(len(families))
    w = 0.27
    for i, m in enumerate(methods):
        ys = [results[f].get(m, {}).get("mean_jaccard", 0) * 100 for f in families]
        ax.bar(x + (i - 1) * w, ys, width=w, color=COLORS[m], label=LABELS[m],
               edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in families])
    ax.set_ylabel("Jaccard (%)")
    ax.set_title("Review Jaccard across LLM families", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="none", fontsize=7)
    ax.grid(axis="y", ls="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def plot_component_ablation(review_abl: dict, output_path: str) -> None:
    """Bar chart of Jaccard / Stability across decomposition variants."""
    order = ["naive", "schema_only", "schema_retry", "ours_full"]
    labels = ["naive", "schema-only", "schema+retry", "ours-full"]
    j_vals  = [review_abl.get(v, {}).get("mean_jaccard", 0) * 100 for v in order]
    s_vals  = [review_abl.get(v, {}).get("mean_stability", 0) * 100 for v in order]

    fig, ax1 = plt.subplots(figsize=(5.5, 2.7))
    x = np.arange(len(order))
    w = 0.36

    bars1 = ax1.bar(x - w/2, j_vals, width=w, color="#5B8DB8",
                    label="Jaccard (%)", edgecolor="black", linewidth=0.4)
    ax1.set_ylabel("Jaccard (%)", color="#345A78")
    ax1.tick_params(axis="y", labelcolor="#345A78")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + w/2, s_vals, width=w, color="#4DAF4A",
                    label="Stability (%)", edgecolor="black", linewidth=0.4)
    ax2.set_ylabel("Stability (%)", color="#2D6F2A")
    ax2.tick_params(axis="y", labelcolor="#2D6F2A")
    ax2.set_ylim(0, 105)
    ax1.set_ylim(0, max(50, max(j_vals) * 1.2 if j_vals else 50))

    ax1.set_title("Component decomposition (review task)", fontweight="bold")
    ax1.grid(axis="y", ls="--", alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    lines = bars1.get_children() + bars2.get_children()
    labels_l = ["Jaccard (%)", "Stability (%)"]
    fig.legend([bars1, bars2], labels_l, loc="upper left",
               bbox_to_anchor=(0.12, 0.98), fontsize=7, framealpha=0.9, edgecolor="none")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def main():
    os.makedirs("figures", exist_ok=True)

    if os.path.exists("results/bbh/results.json"):
        with open("results/bbh/results.json") as f:
            bbh = json.load(f)
        plot_bbh_curves(bbh, "figures/bbh_curves.pdf")
        plot_bbh_curves(bbh, "figures/bbh_curves.png")

    if os.path.exists("results/synthetic/results.json"):
        with open("results/synthetic/results.json") as f:
            syn = json.load(f)
        plot_synthetic_curves(syn, "figures/synthetic_curves.pdf")
        plot_synthetic_curves(syn, "figures/synthetic_curves.png")

    if os.path.exists("results/insurance/results.json"):
        with open("results/insurance/results.json") as f:
            ins = json.load(f)
        plot_insurance_curves(ins, "figures/insurance_curves.pdf")
        plot_insurance_curves(ins, "figures/insurance_curves.png")
        plot_ablation_bar(ins, "figures/ablation_bar.pdf")
        plot_ablation_bar(ins, "figures/ablation_bar.png")

    if os.path.exists("results/review/results.json"):
        with open("results/review/results.json") as f:
            rev = json.load(f)
        plot_review_stability(rev, "figures/review_stability.pdf")
        plot_review_stability(rev, "figures/review_stability.png")

    if os.path.exists("results/review/multi_model.json"):
        with open("results/review/multi_model.json") as f:
            mm = json.load(f)
        plot_multi_model(mm, "figures/multi_model.pdf")
        plot_multi_model(mm, "figures/multi_model.png")

    if os.path.exists("results/review/ablations.json"):
        with open("results/review/ablations.json") as f:
            abl = json.load(f)
        plot_component_ablation(abl, "figures/component_ablation.pdf")
        plot_component_ablation(abl, "figures/component_ablation.png")


if __name__ == "__main__":
    main()
