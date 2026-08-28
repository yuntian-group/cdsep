"""TextGrad-style optimizer for prompt refinement via textual gradients."""

from __future__ import annotations

from cdsep.agent import Agent
from cdsep.graph import ComputationGraph
from cdsep.llm import LLMClient

SEPARATED_OPTIMIZER_PROMPT = """\
You are a prompt optimization expert. Your task is to improve an agent's system prompt \
based on performance feedback from completed episodes.

IMPORTANT: You must ONLY modify the agent's behavioral instructions (the data-flow prompt). \
Do NOT modify, remove, or reference the JSON control schema instructions. \
The control schema format is handled separately and must remain unchanged.

## Current System Prompt
{current_prompt}

## Interaction Summary (data-flow messages only)
{interaction_summary}

## Performance Feedback
Loss/error: {loss}
{feedback}

## Instructions
Based on the feedback above, produce an improved version of the system prompt. \
Focus on:
1. Making instructions clearer and more specific
2. Addressing failure patterns observed in the interaction summary
3. Improving the quality of the agent's reasoning and output

Output ONLY the new system prompt text (no explanations, no JSON, no markdown fences)."""

NAIVE_OPTIMIZER_PROMPT = """\
You are a prompt optimization expert. Your task is to improve an agent's full prompt \
based on performance feedback.

## Current Full Prompt (system message sent to the agent)
{current_full_prompt}

## Full Interaction Trace (including control signals)
{full_trace}

## Performance Feedback
Loss/error: {loss}
{feedback}

## Instructions
Produce an improved version of the full prompt. You may modify any part of it, \
including format instructions, output schemas, or behavioral instructions.

Output ONLY the new prompt text (no explanations, no markdown fences)."""


class TextGradOptimizer:
    """Optimizes agent prompts using textual gradients from an LLM.

    Two modes:
    - separated=True (ours): only edits the data-flow system_prompt
    - separated=False (naive TextGrad): can edit the entire prompt including schema instructions
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def optimize(
        self,
        agent: Agent,
        graph: ComputationGraph,
        loss: float,
        feedback: str = "",
        separated: bool = True,
    ) -> str:
        """Propose an updated prompt for the given agent.

        Args:
            agent: The agent whose prompt to optimize.
            graph: Computation graph with episode traces.
            loss: Scalar loss value.
            feedback: Optional textual feedback.
            separated: If True, only modify data-flow prompt. If False, modify everything.

        Returns:
            The proposed new prompt string.
        """
        if separated:
            interaction_summary = graph.build_interaction_summary(agent.name, include_all=False)
            prompt = SEPARATED_OPTIMIZER_PROMPT.format(
                current_prompt=agent.system_prompt,
                interaction_summary=interaction_summary or "(no interactions recorded)",
                loss=loss,
                feedback=feedback or "Improve overall task performance.",
            )
        else:
            full_trace = graph.build_full_trace_summary()
            prompt = NAIVE_OPTIMIZER_PROMPT.format(
                current_full_prompt=agent.build_system_message(),
                full_trace=full_trace or "(no trace recorded)",
                loss=loss,
                feedback=feedback or "Improve overall task performance.",
            )

        messages = [{"role": "user", "content": prompt}]
        new_prompt = self.llm.chat(messages, temperature=1)
        new_prompt = new_prompt.strip()

        if new_prompt.startswith("```"):
            lines = new_prompt.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            new_prompt = "\n".join(lines).strip()

        return new_prompt

    def optimize_agents(
        self,
        agents: dict[str, Agent],
        graph: ComputationGraph,
        loss: float,
        feedback: str = "",
        separated: bool = True,
    ) -> dict[str, str]:
        """Optimize all agents' prompts and return a dict of new prompts."""
        new_prompts = {}
        for name, agent in agents.items():
            new_prompts[name] = self.optimize(agent, graph, loss, feedback, separated)
        return new_prompts
