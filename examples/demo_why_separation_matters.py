#!/usr/bin/env python3
"""
Demo: Why Control-Data Flow Separation Matters
===============================================

This script runs the SAME multi-agent task three ways and shows exactly
what goes wrong without separation.

Task: A "coordinator" agent decides which of two specialist agents
(math_solver, word_counter) should handle a user question, then the
specialist answers, then the coordinator summarises.

We show:
  1. NAIVE  -- the optimizer can edit everything, including routing fields
              → it often BREAKS the pipeline (bad JSON, wrong agent names)
  2. OURS   -- the optimizer can only edit the "how you explain" part
              → routing always works, AND answers get better
  3. FIXED  -- no optimization at all (baseline)

Run:
    export OPENAI_API_KEY=sk-...
    python examples/demo_why_separation_matters.py
"""

from __future__ import annotations

import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import Literal

from cdsep.agent import Agent, AgentOutput
from cdsep.episode import EpisodeTrace, run_episode
from cdsep.graph import ComputationGraph
from cdsep.llm import LLMClient
from cdsep.optimizer import TextGradOptimizer

# ---------------------------------------------------------------------------
# 1. Define control schemas  (these are the "types" that Python routing uses)
# ---------------------------------------------------------------------------

@dataclass
class CoordinatorControl:
    """The coordinator picks which specialist to call, or stops."""
    route_to: Literal["math_solver", "word_counter", "done"]

@dataclass
class SpecialistControl:
    """Each specialist just signals it's finished."""
    answer: str

# ---------------------------------------------------------------------------
# 2. Define the agents
# ---------------------------------------------------------------------------

COORDINATOR_PROMPT = """\
You coordinate between two specialists. Given a user question, decide:
- "math_solver" if it involves arithmetic, math, or numbers
- "word_counter" if it involves counting words, letters, or text analysis
- "done" if you already have the final answer from a specialist

When routing, explain WHY you chose that specialist.
When finishing, summarise the specialist's answer."""

MATH_PROMPT = """\
You are a math specialist. Solve the given math problem step by step.
Put your final numerical answer in the "answer" field."""

WORD_PROMPT = """\
You are a word/text analysis specialist. Answer the given text question.
Put your final answer in the "answer" field."""


def make_agents(coord_prompt=None, math_prompt=None, word_prompt=None):
    return {
        "coordinator": Agent("coordinator", CoordinatorControl,
                             coord_prompt or COORDINATOR_PROMPT),
        "math_solver": Agent("math_solver", SpecialistControl,
                             math_prompt or MATH_PROMPT),
        "word_counter": Agent("word_counter", SpecialistControl,
                              word_prompt or WORD_PROMPT),
    }


def route(control):
    if isinstance(control, CoordinatorControl):
        if control.route_to == "done":
            return "terminate"
        return control.route_to
    if isinstance(control, SpecialistControl):
        return "coordinator"
    return "terminate"


# ---------------------------------------------------------------------------
# 3. Test questions with ground truth
# ---------------------------------------------------------------------------

QUESTIONS = [
    {"q": "What is 17 * 23?", "answer": "391", "type": "math"},
    {"q": "How many words are in 'the quick brown fox jumps'?", "answer": "5", "type": "word"},
    {"q": "What is 144 / 12 + 7?", "answer": "19", "type": "math"},
    {"q": "How many vowels in 'encyclopedia'?", "answer": "6", "type": "word"},
    {"q": "What is 2^10?", "answer": "1024", "type": "math"},
    {"q": "How many letters in 'Mississippi'?", "answer": "11", "type": "word"},
]


def eval_one(agents, question, llm):
    """Run one question, return (correct, stable, predicted, trace)."""
    trace = run_episode(
        entry_agent=agents["coordinator"],
        agents=agents,
        route_fn=route,
        task_input=question["q"],
        llm=llm,
        max_steps=6,
    )
    stable = trace.is_stable and trace.outcome == "completed"
    predicted = None
    if stable and len(trace.steps) >= 2:
        for step in reversed(trace.steps):
            if hasattr(step.control, "answer"):
                predicted = step.control.answer
                break
    correct = predicted is not None and question["answer"].lower() in str(predicted).lower()
    return correct, stable, predicted, trace


def run_demo():
    llm = LLMClient(model="gpt-5.4-nano", temperature=0)
    opt_llm = LLMClient(model="gpt-5.4-mini", temperature=1)

    # ── FIXED baseline ────────────────────────────────────────────
    print("=" * 70)
    print("  METHOD 1: FIXED PROMPTS (no optimisation)")
    print("=" * 70)
    agents = make_agents()
    fixed_correct, fixed_stable = 0, 0
    for q in QUESTIONS:
        ok, stable, pred, _ = eval_one(agents, q, llm)
        fixed_correct += ok
        fixed_stable += stable
        status = "✓" if ok else "✗"
        print(f"  {status} Q: {q['q']:<45} expected={q['answer']:<6} got={pred}")
    print(f"\n  Accuracy: {fixed_correct}/{len(QUESTIONS)}   "
          f"Stability: {fixed_stable}/{len(QUESTIONS)}")

    # ── Now run 3 optimisation iterations ─────────────────────────
    optimizer = TextGradOptimizer(opt_llm)

    for method_name, separated in [("NAIVE TextGrad (no separation)", False),
                                    ("OURS  (control-data separated)", True)]:
        print(f"\n{'=' * 70}")
        print(f"  METHOD: {method_name}  --  3 optimisation steps")
        print("=" * 70)

        # Reset prompts
        current = {
            "coordinator": COORDINATOR_PROMPT,
            "math_solver": MATH_PROMPT,
            "word_counter": WORD_PROMPT,
        }

        for iteration in range(3):
            agents = make_agents(current["coordinator"], current["math_solver"], current["word_counter"])
            graph = ComputationGraph()
            batch_correct, batch_total = 0, 0

            # Train on first 4 questions
            for q in QUESTIONS[:4]:
                ok, stable, pred, trace = eval_one(agents, q, llm)
                graph.add_from_trace(trace, current)
                batch_correct += ok
                batch_total += 1

            loss = 1.0 - batch_correct / batch_total
            feedback = f"Accuracy this batch: {batch_correct}/{batch_total}"

            for name in current:
                new_p = optimizer.optimize(agents[name], graph, loss,
                                           feedback=feedback, separated=separated)
                current[name] = new_p

            print(f"  iter {iteration}: train_acc={batch_correct}/{batch_total}, loss={loss:.2f}")

        # Final eval on ALL questions
        agents = make_agents(current["coordinator"], current["math_solver"], current["word_counter"])
        total_correct, total_stable = 0, 0
        for q in QUESTIONS:
            ok, stable, pred, trace = eval_one(agents, q, llm)
            total_correct += ok
            total_stable += stable
            status = "✓" if ok else "✗"
            crash = "" if stable else " [UNSTABLE]"
            print(f"  {status} Q: {q['q']:<45} expected={q['answer']:<6} got={pred}{crash}")
        print(f"\n  Accuracy:  {total_correct}/{len(QUESTIONS)}   "
              f"Stability: {total_stable}/{len(QUESTIONS)}")

        if not separated:
            n_unstable = len(QUESTIONS) - total_stable
            if n_unstable:
                print(f"\n  ⚠  {n_unstable} episodes CRASHED due to broken control signals!")
                print("     (This is exactly the problem control-data separation solves.)")

    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print("=" * 70)
    print(textwrap.dedent("""\
        Fixed prompts:  Baseline -- works but not optimised.
        Naive TextGrad: Optimizer can change routing instructions →
                        pipeline often breaks (parse errors, wrong agent names).
        Ours:           Optimizer can ONLY change data-flow prompts →
                        routing always works, AND quality improves.

        This is the core insight of control-data flow separation:
        Keep the structured control channel (routing, termination, schemas)
        FROZEN, and only optimise the free-text reasoning channel.
    """))


if __name__ == "__main__":
    run_demo()
