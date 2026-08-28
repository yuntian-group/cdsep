#!/usr/bin/env python3
"""02 - Multi-agent routing: leader picks one of two specialists.

Demonstrates the two key cdsep guarantees in action:
  (a) the leader's ``target`` field is a Literal, so it can only ever name a
      real specialist; the optimizer cannot route to a nonexistent agent;
  (b) routing is a Python function over the typed control object -- not over
      free-form text -- so format drift cannot break execution.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/02_multi_agent_routing.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cdsep import Agent, LLMClient, run_episode


@dataclass
class CoordinatorControl:
    """Coordinator chooses a specialist or stops."""
    route_to: Literal["math_solver", "word_counter", "done"]


@dataclass
class SpecialistControl:
    """Each specialist returns its final answer as a string."""
    answer: str


def make_agents() -> dict[str, Agent]:
    return {
        "coordinator": Agent(
            name="coordinator",
            control_schema=CoordinatorControl,
            system_prompt=(
                "You receive a user question and dispatch it to a specialist. "
                "Pick math_solver for arithmetic, word_counter for text "
                "questions, or done if you already have a final answer."
            ),
        ),
        "math_solver": Agent(
            name="math_solver",
            control_schema=SpecialistControl,
            system_prompt="You are a math specialist. Solve the problem step by step.",
        ),
        "word_counter": Agent(
            name="word_counter",
            control_schema=SpecialistControl,
            system_prompt="You are a text/word analysis specialist.",
        ),
    }


def route(control) -> str:
    """Pure-Python routing over typed controls. No string parsing involved."""
    if isinstance(control, CoordinatorControl):
        if control.route_to == "done":
            return "terminate"
        return control.route_to
    if isinstance(control, SpecialistControl):
        return "coordinator"
    return "terminate"


def main() -> None:
    agents = make_agents()
    llm = LLMClient(model="gpt-5.4-nano", temperature=0)

    trace = run_episode(
        entry_agent=agents["coordinator"],
        agents=agents,
        route_fn=route,
        task_input="How many vowels are in the word 'encyclopedia'?",
        llm=llm,
        max_steps=6,
    )

    print(f"Outcome:    {trace.outcome}")
    print(f"Stable:     {trace.is_stable}  (protocol guarantees: True by design)")
    for s in trace.steps:
        print(f"  step {s.step:>2}  [{s.agent_name:<13}]  control={s.control}")
    print(f"Final message: {trace.final_message[:200]}")


if __name__ == "__main__":
    main()
