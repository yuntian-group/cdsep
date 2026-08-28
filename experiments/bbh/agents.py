"""Agent definitions for BBH tasks.

Because BBH tasks have heterogeneous answer spaces (multiple-choice letters,
Yes/No, free-form strings), we synthesise a per-task control schema at
runtime. The schema field is always called ``answer`` so the optimizer prompt
templates remain task-agnostic.
"""

from __future__ import annotations

from dataclasses import make_dataclass
from typing import Any, Literal, Type

from cdsep.agent import Agent


def make_control_schema(meta: dict) -> Type:
    """Construct a dataclass type with the right ``answer`` annotation.

    For multiple-choice tasks the ``answer`` field is a ``Literal`` over the
    valid option letters, so the schema VALIDATOR will reject any other value.
    """
    kind = meta["kind"]
    if kind == "choice":
        opts = tuple(meta["options"])
        ann = Literal[opts]  # type: ignore[valid-type]
    elif kind == "yesno":
        ann = Literal["Yes", "No"]
    else:
        ann = str
    return make_dataclass(
        f"BBHAnswer_{meta['task']}",
        [("answer", ann)],
    )


SEP_PROMPT_TEMPLATE = """\
You are a careful reasoning assistant solving tasks from the BIG-Bench Hard
benchmark. Read each question carefully, think step by step, and produce the
correct final answer.

Examples:

{few_shot}

Now answer the next question.

Reasoning protocol:
1. First, write your step-by-step reasoning in plain text. Enumerate
   intermediate states, candidate options, and any constraints from the
   question.
2. After the reasoning, end your response with the JSON control block
   containing exactly the final answer."""


NAIVE_PROMPT_TEMPLATE = """\
You are a careful reasoning assistant solving tasks from the BIG-Bench Hard
benchmark. Read each question carefully, think step by step, and produce the
correct final answer.

Examples:

{few_shot}

Output format: respond with a JSON object on the first line of the form
{{"answer": "<your answer>"}} (for multiple-choice tasks the answer should be
a single letter such as "A"; for Yes/No tasks the answer should be exactly
"Yes" or "No"). After the JSON, you may include a short explanation.

Now answer the next question."""


def make_bbh_agent(
    few_shot_text: str,
    meta: dict,
    prompt: str | None = None,
    separated: bool = True,
    json_position: str = "end",
) -> Agent:
    schema_cls = make_control_schema(meta)
    template = prompt or (SEP_PROMPT_TEMPLATE if separated else NAIVE_PROMPT_TEMPLATE)
    system_prompt = template.format(few_shot=few_shot_text)
    return Agent(
        name="solver",
        control_schema=schema_cls,
        system_prompt=system_prompt,
        separated=separated,
        json_position=json_position if separated else "begin",
    )
