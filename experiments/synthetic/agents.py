"""Agent definitions for synthetic function induction."""

from __future__ import annotations

from dataclasses import dataclass

from cdsep.agent import Agent


@dataclass
class SyntheticControl:
    answer: int


INITIAL_PROMPT = """\
You are given examples of a hidden function f(a, b) that takes two integers and returns an integer.
Study the examples below to discover the pattern, then answer the query.

{examples}

Answer the following query. Output ONLY the integer result in the JSON control block."""


# Initial prompt for the NAIVE baseline (no auto-scaffolding).
# Includes inline format instructions that the optimizer is free to modify or remove.
NAIVE_INITIAL_PROMPT = """\
You are given examples of a hidden function f(a, b) that takes two integers and returns an integer.
Study the examples below to discover the pattern, then answer the query.

{examples}

Output format: respond with a JSON object on the first line with a single field
"answer" containing the integer result, then optionally followed by an explanation.
Example: {{"answer": 7}}

Answer the following query."""


def make_synthetic_agent(
    examples_text: str,
    prompt: str | None = None,
    separated: bool = True,
) -> Agent:
    """Create the single-agent solver for synthetic tasks."""
    if prompt is None:
        prompt = INITIAL_PROMPT if separated else NAIVE_INITIAL_PROMPT
    system_prompt = prompt.format(examples=examples_text)
    return Agent(
        name="solver",
        control_schema=SyntheticControl,
        system_prompt=system_prompt,
        separated=separated,
    )
