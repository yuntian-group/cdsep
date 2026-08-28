"""Agent definitions for collaborative review generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cdsep.agent import Agent


@dataclass
class LeaderControl:
    action: Literal["send", "stop"]
    target_agent: Literal["worker_1", "worker_2", "worker_3", "none"]
    stop: bool


@dataclass
class WorkerControl:
    status: Literal["done"]
    section: str


LEADER_PROMPT = """\
You are the leader of a paper review team with 3 workers: worker_1, worker_2, worker_3.

Your job:
1. Assign each worker a DIFFERENT section of the paper (introduction, methods, experiments).
2. After all workers report, synthesize their comments into a final review.
3. Produce a list of atomic review comments (each making exactly one point).

Process:
- First, send assignments to each worker one at a time.
- After receiving all worker responses, set stop=true and output the final merged review comments.

In your message, either give the assignment (which section to review) or provide the final merged comments."""

WORKER_PROMPT = """\
You are a paper reviewer assigned to review a specific section.

Given the section text, produce 3-5 atomic review comments. Each comment should:
- Address exactly one point (clarity, correctness, significance, or missing element)
- Be concise (1-2 sentences)
- Be specific to the content

In the "section" field of your control block, state which section you reviewed.
In your message, list your atomic comments as a numbered list."""


NAIVE_LEADER_PROMPT = """\
You are the leader of a paper review team with 3 workers: worker_1, worker_2, worker_3.

Your job:
1. Assign each worker a DIFFERENT section of the paper.
2. After all workers report, synthesize their comments into a final review.
3. Produce a list of atomic review comments.

Output format: respond with a JSON object on the first line with these fields:
- action: either "send" or "stop"
- target_agent: one of "worker_1", "worker_2", "worker_3", or "none"
Example: {"action": "send", "target_agent": "worker_1"}

After the JSON, write either the assignment for the next worker or, if stopping,
the final merged review comments as a numbered list."""

NAIVE_WORKER_PROMPT = """\
You are a paper reviewer assigned to review a specific section.

Output format: respond with a JSON object on the first line with these fields:
- status: "done"
- section: short name of the section you reviewed
Example: {"status": "done", "section": "introduction"}

After the JSON, list 3-5 atomic review comments as a numbered list."""


def make_review_agents(
    leader_prompt: str | None = None,
    worker_prompt: str | None = None,
    separated: bool = True,
) -> dict[str, Agent]:
    if separated:
        leader_default = LEADER_PROMPT
        worker_default = WORKER_PROMPT
    else:
        leader_default = NAIVE_LEADER_PROMPT
        worker_default = NAIVE_WORKER_PROMPT

    agents = {
        "leader": Agent(
            name="leader",
            control_schema=LeaderControl,
            system_prompt=leader_prompt or leader_default,
            separated=separated,
        ),
    }
    for i in range(1, 4):
        agents[f"worker_{i}"] = Agent(
            name=f"worker_{i}",
            control_schema=WorkerControl,
            system_prompt=worker_prompt or worker_default,
            separated=separated,
        )
    return agents
