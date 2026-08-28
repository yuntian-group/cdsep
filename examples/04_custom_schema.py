#!/usr/bin/env python3
"""04 - Custom schemas: Pydantic models, Optionals, and nested controls.

cdsep accepts any dataclass *or* Pydantic model as a control schema. Pydantic
gives you Optional fields, validators, and nested objects -- and the schema
scaffolding is still auto-generated.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/04_custom_schema.py
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from cdsep import Agent, LLMClient, generate_scaffolding, run_single_agent_episode


class Citation(BaseModel):
    """A nested object: every cited paper has a year and an author."""
    year: int = Field(ge=1800, le=2100)
    author: str


class ResearchSummary(BaseModel):
    """Pydantic control schema with Optional and nested fields."""
    topic: Literal["nlp", "vision", "rl", "theory", "other"]
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=10)
    primary_citation: Optional[Citation] = None


def main() -> None:
    print("Auto-generated scaffolding:")
    print("-" * 60)
    print(generate_scaffolding(ResearchSummary))
    print("-" * 60)

    agent = Agent(
        name="summarizer",
        control_schema=ResearchSummary,
        system_prompt=(
            "You are a research summarizer. Read the abstract and produce a "
            "structured summary. Always estimate your confidence."
        ),
    )

    abstract = (
        "We propose a new attention mechanism for transformers that reduces "
        "complexity from quadratic to linear by using learned routing. "
        "Experiments on long-document summarization show competitive quality "
        "at 4x lower memory."
    )
    llm = LLMClient(model="gpt-5.4-nano", temperature=0)
    trace = run_single_agent_episode(agent, abstract, llm)

    if trace.is_stable and trace.steps[0].control is not None:
        c = trace.steps[0].control
        print(f"\nTopic:      {c.topic}")
        print(f"Confidence: {c.confidence:.2f}")
        print(f"Summary:    {c.summary}")
        if c.primary_citation:
            print(f"Citation:   {c.primary_citation.author} ({c.primary_citation.year})")
    else:
        print("\nAgent failed; errors:", trace.steps[0].errors)


if __name__ == "__main__":
    main()
