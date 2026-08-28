"""DSPy baseline for the MARG review-generation experiment.

Implements the same leader-worker pipeline as our framework, but built with
DSPy primitives and optimized via DSPy's own teleprompters (BootstrapFewShot
and MIPROv2). Uses the same train/val/test split, the same gpt-5.4-nano
model, and the same MARG alignment-based metric so the results are directly
comparable to ``tab:review``.

We emulate three workers as one DSPy module that produces three sets of
section-specific comments (a faithful translation of the leader-worker
structure into DSPy's signature-based world). DSPy's own format guards are
used; we report stability as the fraction of papers where the pipeline
produces a non-empty list of valid string comments. Because DSPy uses its
own JSON / "ChainOfThought" parsing logic, stability here is *not*
guaranteed by construction (unlike ours), and falls into the same regime
as the optimized prompts that DSPy's teleprompters produce.
"""

from __future__ import annotations

import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

warnings.filterwarnings("ignore")

import dspy
from dspy.teleprompt import BootstrapFewShot, MIPROv2

from cdsep.llm import LLMClient
from experiments.review.configs import (
    MODEL,
    N_PAPERS,
    N_TRAIN,
    N_TEST,
    N_VAL,
    OPTIMIZER_MODEL,
    SEEDS,
)
from experiments.review.marg_data import load_papers as load_marg_papers
from experiments.review.marg_metrics import compute_metrics_marg
from experiments.review.run import format_paper_input, split_three_ways


# ---------------------------------------------------------------------------
# DSPy signatures
# ---------------------------------------------------------------------------

class WorkerReview(dspy.Signature):
    """Read a section of a scientific paper and write 3-5 atomic review
    comments. Each comment should make exactly one substantive point about
    clarity, correctness, significance, or a missing element.
    """

    paper_section: str = dspy.InputField(desc="One section of a scientific paper.")
    section_name: str = dspy.InputField(desc="Name of the section (e.g. methods).")
    comments: list[str] = dspy.OutputField(desc="3-5 atomic review comments.")


class LeaderAggregate(dspy.Signature):
    """Merge three workers' atomic review comments into a single
    de-duplicated, atomic-comment review that covers the whole paper.
    """

    title: str = dspy.InputField()
    abstract: str = dspy.InputField()
    worker1_comments: list[str] = dspy.InputField()
    worker2_comments: list[str] = dspy.InputField()
    worker3_comments: list[str] = dspy.InputField()
    final_comments: list[str] = dspy.OutputField(
        desc="A merged, de-duplicated list of atomic review comments."
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ReviewPipeline(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.worker = dspy.ChainOfThought(WorkerReview)
        self.leader = dspy.ChainOfThought(LeaderAggregate)

    def forward(self, paper: dict) -> list[str]:
        sections = list(paper.get("sections", {}).items())
        # Distribute sections across 3 workers
        if not sections:
            return dspy.Prediction(comments=[], worker_ok=[False, False, False], leader_ok=False)
        chunks = [[], [], []]
        for i, (name, text) in enumerate(sections):
            chunks[i % 3].append((name, text))

        worker_outputs: list[list[str]] = []
        worker_ok: list[bool] = []
        for chunk in chunks:
            if not chunk:
                worker_outputs.append([])
                worker_ok.append(False)
                continue
            joined_text = "\n\n".join(f"## {n}\n{t}" for n, t in chunk)
            joined_name = ", ".join(n for n, _ in chunk)
            try:
                pred = self.worker(paper_section=joined_text, section_name=joined_name)
                comments = pred.comments if isinstance(pred.comments, list) else []
            except Exception:
                comments = []
            cleaned = [str(c) for c in comments if isinstance(c, (str, int, float)) and str(c).strip()][:5]
            worker_outputs.append(cleaned)
            worker_ok.append(len(cleaned) >= 1)

        leader_ok = False
        try:
            agg = self.leader(
                title=paper.get("title", ""),
                abstract=paper.get("abstract", ""),
                worker1_comments=worker_outputs[0],
                worker2_comments=worker_outputs[1],
                worker3_comments=worker_outputs[2],
            )
            comments = agg.final_comments if isinstance(agg.final_comments, list) else []
            cleaned = [str(c) for c in comments if isinstance(c, (str, int, float)) and str(c).strip()][:8]
            leader_ok = len(cleaned) >= 1
        except Exception:
            cleaned = []

        return dspy.Prediction(comments=cleaned, worker_ok=worker_ok, leader_ok=leader_ok)


# ---------------------------------------------------------------------------
# Adapter: cdsep LLMClient -> DSPy LM
# ---------------------------------------------------------------------------

def _configure_dspy(model: str = MODEL):
    """Configure DSPy with a free-form ChatAdapter (NOT JSON mode).

    DSPy 3.x's default JSONAdapter tries to use the provider's structured-
    output API (OpenAI's response_format=json_schema, etc). For gpt-5.4-nano
    that path is fragile, so we explicitly use ChatAdapter which simply asks
    the model to produce a textual format DSPy can parse.

    DSPy 3.2 routes through litellm; litellm's hard-coded gpt-5 metadata is
    out of date for gpt-5.4 (it thinks temperature is unsupported), so we
    set ``drop_params`` to silently drop the offending kwargs.
    """
    import litellm
    litellm.drop_params = True

    lm = dspy.LM(
        model=f"openai/{model}",
        cache=True,
        temperature=1,
        max_tokens=4096,
    )
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter())
    return lm


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def _make_metric_fn(eval_llm: LLMClient):
    def metric(example, pred, trace=None):
        comments = list(pred.comments) if hasattr(pred, "comments") else []
        m = compute_metrics_marg(comments, example.human_reviews, eval_llm)
        return float(m["jaccard"])
    return metric


def _to_examples(papers: list[dict]) -> list:
    """Convert papers into dspy.Example objects."""
    return [
        dspy.Example(
            paper=p,
            human_reviews=p["human_reviews"],
        ).with_inputs("paper")
        for p in papers
    ]


def _evaluate(pipeline: ReviewPipeline, papers: list[dict], eval_llm: LLMClient) -> dict:
    """Evaluate DSPy review pipeline.

    Stability (lenient): pipeline returned a non-empty list of comments.
    Strict stability: every internal stage (all 3 workers + leader merge) emitted
    at least one non-empty string. This is the closest analogue to Ours'
    schema-level validation; the collapsed DSPy pipeline has no routing
    decision to validate, so we validate per-stage non-empty outputs instead.
    """
    rs, ps, js, stab, strict_stab = [], [], [], 0, 0
    n_comments_total = 0
    for paper in papers:
        worker_ok: list[bool] = []
        leader_ok = False
        try:
            pred = pipeline(paper=paper)
            comments = list(pred.comments) if hasattr(pred, "comments") else []
            worker_ok = list(getattr(pred, "worker_ok", []))
            leader_ok = bool(getattr(pred, "leader_ok", False))
        except Exception:
            comments = []
        if comments:
            stab += 1
        # Strict: all 3 workers AND the leader produced >=1 valid comment.
        if leader_ok and len(worker_ok) >= 3 and all(worker_ok[:3]):
            strict_stab += 1
        n_comments_total += len(comments)
        m = compute_metrics_marg(comments, paper["human_reviews"], eval_llm)
        rs.append(m["recall"])
        ps.append(m["precision"])
        js.append(m["jaccard"])
    n = max(1, len(papers))
    return {
        "recall": sum(rs) / n,
        "precision": sum(ps) / n,
        "jaccard": sum(js) / n,
        "n_comments": n_comments_total / n,
        "stability": stab / n,
        "strict_stability": strict_stab / n,
    }


# ---------------------------------------------------------------------------
# One run per (compiler, seed)
# ---------------------------------------------------------------------------

def run_one(compiler_name: str, seed: int) -> dict:
    print(f"\n--- DSPy {compiler_name} / seed {seed} ---", flush=True)
    papers = load_marg_papers(seed=seed, limit=N_PAPERS)
    train, val, test = split_three_ways(papers, N_TRAIN, N_VAL)
    eval_llm = LLMClient(model=MODEL, temperature=0)

    _configure_dspy(MODEL)

    pipeline = ReviewPipeline()
    metric_fn = _make_metric_fn(eval_llm)

    # Compile (= optimize)
    train_ex = _to_examples(train)
    val_ex = _to_examples(val)

    if compiler_name == "bootstrap_fewshot":
        compiler = BootstrapFewShot(metric=metric_fn, max_bootstrapped_demos=2,
                                     max_labeled_demos=2, max_rounds=1)
        try:
            compiled = compiler.compile(pipeline, trainset=train_ex)
        except Exception as e:
            print(f"  BootstrapFewShot compile failed: {e}", flush=True)
            compiled = pipeline
    elif compiler_name == "miprov2":
        compiler = MIPROv2(metric=metric_fn, auto="light", num_threads=1)
        try:
            compiled = compiler.compile(
                pipeline,
                trainset=train_ex,
                valset=val_ex,
                requires_permission_to_run=False,
            )
        except Exception as e:
            print(f"  MIPROv2 compile failed: {e}", flush=True)
            compiled = pipeline
    elif compiler_name == "no_compile":
        compiled = pipeline
    else:
        raise ValueError(compiler_name)

    test_metrics = _evaluate(compiled, test, eval_llm)
    print(f"  -> R={test_metrics['recall']:.3f} P={test_metrics['precision']:.3f} "
          f"J={test_metrics['jaccard']:.3f} stab={test_metrics['stability']:.3f} "
          f"strict={test_metrics.get('strict_stability', 0):.3f}",
          flush=True)
    return test_metrics


def main() -> None:
    os.makedirs("results/review", exist_ok=True)
    out: dict[str, dict] = {}

    for compiler in ("no_compile", "bootstrap_fewshot", "miprov2"):
        seed_results = []
        for seed in SEEDS:
            try:
                r = run_one(compiler, seed)
            except Exception as e:
                print(f"  seed {seed} CRASHED: {e}", flush=True)
                r = {"recall": 0, "precision": 0, "jaccard": 0, "stability": 0,
                     "n_comments": 0}
            seed_results.append(r)

        keys = ["recall", "precision", "jaccard", "n_comments", "stability", "strict_stability"]
        avgs = {f"mean_{k}": round(sum(r.get(k, 0) for r in seed_results) / len(seed_results), 4)
                for k in keys}
        out[compiler] = {**avgs, "per_seed": seed_results}
        print(f"\n[{compiler}] AVG: R={avgs['mean_recall']:.3f} "
              f"P={avgs['mean_precision']:.3f} J={avgs['mean_jaccard']:.3f} "
              f"stab={avgs['mean_stability']:.3f}", flush=True)

        with open("results/review/dspy.json", "w") as f:
            json.dump(out, f, indent=2, default=str)

    print("\n\nDSPy SUMMARY")
    print("=" * 70)
    print(f"{'Compiler':<22} {'Recall':>10} {'Precision':>10} {'Jaccard':>10} {'Stab':>10}")
    print("-" * 70)
    for c, d in out.items():
        print(f"{c:<22} {d['mean_recall']:>10.3f} {d['mean_precision']:>10.3f} "
              f"{d['mean_jaccard']:>10.3f} {d['mean_stability']:>10.3f}")


if __name__ == "__main__":
    main()
