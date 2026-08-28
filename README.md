# cdsep — Control–Data Flow Separation for Multi-Agent LLM Optimization

`cdsep` is a Python library for building **multi-agent LLM systems** with
separate **control flow** (routing, termination, and schemas) and **data flow**
(free-form natural language). In separated mode, prompt optimizers edit only
the data-flow prompt; schema instructions remain frozen, and invalid control is
rejected before routing.

This is the reference implementation for the paper
*Control-Data Flow Separation: Stable Prompt Optimization in Multi-Agent LLMs*
(Findings of EMNLP 2026).

## Why?

Naive prompt optimization can edit format and routing instructions together
with task instructions. On the MARG review-generation task, this causes every
evaluation episode to fail after optimization, while the separated
configuration retains 100 % stability and improves Jaccard:

| Method                  | Stability | Jaccard |
|-------------------------|----------:|--------:|
| Fixed prompts           |     100 % |    31.0 |
| Naive TextGrad          |       0 % |     0.0 |
| **Ours (separated)**    | **100 %** |    44.4 |

`cdsep` keeps schema scaffolding in a frozen prompt slot that the optimizer
cannot edit, and Python routing functions consume validated control objects
rather than unstructured messages.

## Install

```bash
pip install -e .                    # core library
pip install -e ".[experiments]"     # library and experiment dependencies
```

Requires Python ≥ 3.10. Set `OPENAI_API_KEY` in your environment before running
anything that calls an LLM.

## Quick start

```python
from dataclasses import dataclass
from typing import Literal
from cdsep import Agent, run_episode, LLMClient

@dataclass
class LeaderControl:
    action: Literal["send", "stop"]
    target: Literal["worker_1", "worker_2", "worker_3"]

leader = Agent(
    name="leader",
    control_schema=LeaderControl,
    system_prompt="You coordinate a review team of three workers ...",
)
worker_1 = Agent(name="worker_1", control_schema=..., system_prompt="...")
# (define worker_2, worker_3 similarly)

def route(ctrl: LeaderControl) -> str:
    return "terminate" if ctrl.action == "stop" else ctrl.target

llm = LLMClient(model="gpt-5.4-nano")
trace = run_episode(
    entry_agent=leader,
    agents={"leader": leader, "worker_1": worker_1, ...},
    route_fn=route,
    task_input=paper_text,
    llm=llm,
)

assert trace.is_stable                # protocol-valid execution trace
```

## Core abstractions

| Object | What it is |
|---|---|
| **`Agent`** | LLM call wrapped with a typed control schema and an optimizable system prompt. |
| **Control schema** | A Python `dataclass` (or Pydantic `BaseModel`) describing the structured fields the agent must produce. |
| **`run_episode`** | Multi-agent loop with a developer-supplied routing function. |
| **`EpisodeTrace`** | Structured trace of `(control, message)` pairs, with stability metadata. |
| **`ComputationGraph`** | DAG over a trace, used to assemble per-agent feedback for the optimizer. |
| **`TextGradOptimizer`** | TextGrad-style optimizer; `separated=True` is the safe default, `separated=False` reproduces the unsafe baseline. |
| **`LLMClient`** | Provider-agnostic client with retry, cost tracking, and disk-backed caching of deterministic calls. |

## Examples

- [`examples/01_quickstart.py`](examples/01_quickstart.py) — minimal single-agent.
- [`examples/02_multi_agent_routing.py`](examples/02_multi_agent_routing.py) — leader / worker pattern.
- [`examples/03_textgrad_optimization.py`](examples/03_textgrad_optimization.py) — full optimization loop.
- [`examples/04_custom_schema.py`](examples/04_custom_schema.py) — Pydantic schemas, nested controls, repair callbacks.
- [`examples/notebooks/walkthrough.ipynb`](examples/notebooks/walkthrough.ipynb) — Jupyter walkthrough.
- [`examples/demo_why_separation_matters.py`](examples/demo_why_separation_matters.py) — runs three methods side-by-side
  and shows naive crashing.

## Reproducing the paper

```bash
# Clone the repository
git clone https://github.com/yuntian-group/cdsep.git
cd cdsep && pip install -e ".[experiments]"

# Re-run each experiment (uses ~/.cdsep_cache so reruns hit cache)
python experiments/bbh/run.py
python experiments/insurance/run.py
python experiments/review/run.py

# Generate figures + tables
python experiments/generate_figures.py
python experiments/analysis/failure_breakdown.py
python experiments/analysis/prompt_diff.py

# Compile the paper
pdflatex acl_latex.tex && bibtex acl_latex && pdflatex acl_latex.tex && pdflatex acl_latex.tex
```

The MARG/ARIES review benchmark is downloaded from the public S3 mirror
(`ai2-s2-research-public/aries/`) the first time `experiments/review/run.py`
is invoked. The BBH benchmark is fetched via 🤗 Datasets.

The public repository includes the synthetic underwriting benchmark. The
industry-verified underwriting inputs and manual are partner-supplied and are
not redistributed; reproducing those rows requires separately authorized
access. All partner-supplied inputs used in the paper are fully synthetic and
contain no real customer or patient information.

## Repository layout

```
cdsep/                 # core library (pip-installable)
  schema.py            # auto-scaffolding, parsing, validation
  agent.py             # Agent abstraction (separated / naive modes)
  episode.py           # multi-agent episode runner + stability metadata
  graph.py             # computation graph for trace propagation
  optimizer.py         # TextGrad optimizer (separated + naive)
  llm.py               # OpenAI client with caching
  logging_utils.py     # JSONL experiment logging

experiments/           # paper experiments
  bbh/                 # Big-Bench Hard subset (single-agent reasoning)
  insurance/           # 3-agent insurance rating workflow
  review/              # leader-worker scientific review generation (MARG)
  analysis/            # failure-mode and prompt-diff analyses

tests/                 # unit and integration tests
sections/              # paper LaTeX (split per-section)
figures/               # publication-quality PDFs
```

## License

MIT. See [LICENSE](LICENSE).

## Citing

```bibtex
@inproceedings{cdsep2026,
  title     = {Control-Data Flow Separation: Stable Prompt Optimization in Multi-Agent LLMs},
  author    = {Zhang, Wentao and Murtaza, Shariyar and Bhatti, Junaid and
               Soni, Utkarsh and Nie, Yifan and Wen, Eugene and Deng, Yuntian},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026}
}
```
