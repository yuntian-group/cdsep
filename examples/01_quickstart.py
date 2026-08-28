#!/usr/bin/env python3
"""01 - Quickstart: a single-agent solver in 20 lines.

This script defines a one-field control schema, wraps it in an ``Agent``,
and asks the agent to solve a single arithmetic problem.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/01_quickstart.py
"""

from __future__ import annotations

from dataclasses import dataclass

from cdsep import Agent, LLMClient, run_single_agent_episode


@dataclass
class Answer:
    """A typed integer answer. The schema is the control surface."""
    answer: int


def main() -> None:
    agent = Agent(
        name="solver",
        control_schema=Answer,
        system_prompt="You are an arithmetic solver. Compute the integer result.",
    )
    llm = LLMClient(model="gpt-5.4-nano", temperature=0)

    trace = run_single_agent_episode(agent, "What is 17 * 23?", llm)
    step = trace.steps[0]

    print(f"Outcome:    {trace.outcome}")
    print(f"Stable:     {trace.is_stable}")
    print(f"Predicted:  {step.control.answer}")
    print(f"Reasoning:  {step.message[:200]}")


if __name__ == "__main__":
    main()
