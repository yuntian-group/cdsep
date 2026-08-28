"""Agent abstraction: LLM-backed module with control schema and optimizable prompt.

The :class:`Agent` is the unit of computation in cdsep. It wraps a single
LLM call together with:

* a typed *control schema* (a dataclass or Pydantic model) that defines the
  structured part of every output;
* a free-form *system prompt* that the optimizer is allowed to edit;
* (optionally) auto-generated scaffolding that pins the JSON output format
  and is *not* visible to the optimizer.

Agents have two modes:

* ``separated=True`` (default; "ours") - schema scaffolding is auto-generated
  and frozen; failed parses trigger up to ``max_parse_retries`` repair turns;
  protocol stability is guaranteed.
* ``separated=False`` ("naive") - the entire system prompt is a single
  editable string; no auto-scaffolding, no parse retry; this is what we use
  for the unsafe TextGrad baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Type

from cdsep.llm import LLMClient
from cdsep.schema import generate_scaffolding, parse_response, validate_control


@dataclass
class AgentOutput:
    """Result of a single agent call."""
    control: Any
    message: str
    raw: str
    parse_errors: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.parse_errors and not self.validation_errors


class Agent:
    """An LLM-backed agent with a control schema and an optimizable system prompt.

    Two operating modes:

    * ``separated=True`` (default; corresponds to OUR framework):
        - The auto-generated ``schema_prompt`` (control-flow scaffolding) is
          appended to ``system_prompt`` at call time and is FROZEN -- the
          optimizer only ever edits ``system_prompt`` (data flow).
        - On parse/validation failure the agent retries up to
          ``max_parse_retries`` times with explicit error feedback.
        - Provides protocol stability by construction.

    * ``separated=False`` (the NAIVE TextGrad baseline):
        - No auto-generated schema scaffolding. The ENTIRE prompt sent to the
          LLM is just ``system_prompt`` -- whoever (developer or optimizer)
          writes the prompt is responsible for embedding format instructions.
        - No retry on parse/validation failure -- malformed output propagates
          as a failed step (the optimizer is supposed to fix the prompt).
        - This is what happens when developers write multi-agent pipelines
          without a typed control surface.
    """

    def __init__(
        self,
        name: str,
        control_schema: Type,
        system_prompt: str,
        max_parse_retries: int = 2,
        separated: bool = True,
        demo_block: str = "",
        json_position: str = "begin",
    ):
        self.name = name
        self.control_schema = control_schema
        self.system_prompt = system_prompt
        self.demo_block = demo_block
        self.json_position = json_position
        self.schema_prompt = generate_scaffolding(control_schema, json_position=json_position)
        self.max_parse_retries = max_parse_retries if separated else 0
        self.separated = separated

    def build_system_message(self) -> str:
        body = self.system_prompt
        if self.demo_block:
            body = f"{self.demo_block.rstrip()}\n\n## Task instructions\n{body}"
        if self.separated:
            return f"{body}\n\n---\n\n{self.schema_prompt}"
        return body

    def call(
        self,
        context: str,
        llm: LLMClient,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AgentOutput:
        """Call the LLM with the system prompt (+ schema if separated) + context."""
        messages = [{"role": "system", "content": self.build_system_message()}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": context})

        all_parse_errors: list[str] = []
        all_validation_errors: list[str] = []
        last_raw = ""
        last_message = ""

        for attempt in range(1 + self.max_parse_retries):
            raw = llm.chat(messages)
            last_raw = raw
            control_dict, message = parse_response(raw)
            last_message = message

            if control_dict is None:
                all_parse_errors.append(f"Attempt {attempt + 1}: no JSON control block found")
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": "Your response did not contain a valid JSON control block. "
                    "Please try again, starting with the JSON control block.",
                })
                continue

            instance, errors = validate_control(control_dict, self.control_schema)
            if errors:
                all_validation_errors.extend(
                    f"Attempt {attempt + 1}: {e}" for e in errors
                )
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": f"Your control block had validation errors: {errors}. "
                    "Please fix and try again.",
                })
                continue

            return AgentOutput(
                control=instance,
                message=message,
                raw=raw,
            )

        return AgentOutput(
            control=None,
            message=last_message,
            raw=last_raw,
            parse_errors=all_parse_errors,
            validation_errors=all_validation_errors,
        )

    def clone(self) -> Agent:
        """Create a copy with the same schema but independent prompt."""
        return Agent(
            name=self.name,
            control_schema=self.control_schema,
            system_prompt=self.system_prompt,
            max_parse_retries=self.max_parse_retries,
            separated=self.separated,
            demo_block=self.demo_block,
            json_position=self.json_position,
        )
