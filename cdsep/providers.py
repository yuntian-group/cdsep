"""Multi-provider LLM dispatch.

A single :class:`LLMClient` should accept any of:
    OpenAI   (gpt-..., o1-...)
    Anthropic (claude-...)
    Google   (gemini-...)

Each provider has its own SDK and message-format quirks. We hide them behind a
common ``call(model, messages, temperature, **extra)`` -> ``(text, usage)``
function so the rest of the framework (Agent, Episode, Optimizer, cache,
logger) is provider-agnostic.

Message format expected here is the OpenAI chat format:
    [{"role": "system" | "user" | "assistant", "content": str}, ...]
Adapters translate to each provider's native shape.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def detect_provider(model: str) -> str:
    """Return ``"openai" | "anthropic" | "google"`` from a model name."""
    m = model.lower()
    if m.startswith("gpt-") or m.startswith("o1-") or m.startswith("o3-") or m.startswith("o4-"):
        return "openai"
    if m.startswith("claude-"):
        return "anthropic"
    if m.startswith("gemini-") or m.startswith("models/gemini-"):
        return "google"
    raise ValueError(f"Cannot infer provider from model name: {model!r}")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ProviderResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

_openai_client = None


def _openai() -> Any:
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _openai_client


def _is_gpt5(model: str) -> bool:
    return "gpt-5" in model.lower()


def _openai_call(model: str, messages: list[dict], temperature: float, **extra: Any) -> ProviderResult:
    api_kwargs: dict[str, Any] = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        **extra,
    )
    if _is_gpt5(model):
        api_kwargs.setdefault("reasoning_effort", "none")
    resp = _openai().chat.completions.create(**api_kwargs)
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    return ProviderResult(
        text=text,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

_anthropic_client = None


def _anthropic() -> Any:
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        # Try ANTHROPIC_API_KEY first, then fall back to CLAUDE_API_KEY (the
        # name a few of our scripts use locally).
        key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
        _anthropic_client = anthropic.Anthropic(api_key=key)
    return _anthropic_client


def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Anthropic requires the system prompt as a top-level field."""
    system_parts = []
    others = []
    for m in messages:
        if m.get("role") == "system":
            system_parts.append(m.get("content", ""))
        else:
            others.append(m)
    return "\n\n".join(system_parts), others


def _anthropic_call(model: str, messages: list[dict], temperature: float, **extra: Any) -> ProviderResult:
    system_prompt, msgs = _split_system(messages)

    # Anthropic doesn't accept empty content; coerce
    api_msgs = []
    for m in msgs:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        api_msgs.append({"role": role, "content": m.get("content", "") or ""})

    if not api_msgs:
        # If everything was system-only (unlikely but possible), put it in a
        # single user turn so Anthropic accepts the request.
        api_msgs = [{"role": "user", "content": system_prompt or ""}]
        system_prompt = ""

    max_tokens = extra.pop("max_tokens", None) or extra.pop("max_completion_tokens", None) or 4096
    api_kwargs = dict(
        model=model,
        messages=api_msgs,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if system_prompt:
        api_kwargs["system"] = system_prompt
    api_kwargs.update(extra)
    resp = _anthropic().messages.create(**api_kwargs)
    text = ""
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            text += block.text
    usage = getattr(resp, "usage", None)
    return ProviderResult(
        text=text,
        input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
        output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
    )


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------

_google_client = None


def _google() -> Any:
    global _google_client
    if _google_client is None:
        from google import genai
        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("AISTUDIO_API_KEY")
        _google_client = genai.Client(api_key=key)
    return _google_client


def _to_gemini_contents(messages: list[dict]) -> tuple[str, list[dict]]:
    """Convert OpenAI-style messages to Gemini's contents list.

    System messages are gathered into a single ``system_instruction`` string;
    user/assistant messages map to ``user`` / ``model`` roles.
    """
    system_parts = []
    contents = []
    for m in messages:
        role = m.get("role")
        text = m.get("content", "") or ""
        if role == "system":
            system_parts.append(text)
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": text}]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": text}]})
    return "\n\n".join(system_parts), contents


def _google_call(model: str, messages: list[dict], temperature: float, **extra: Any) -> ProviderResult:
    system_prompt, contents = _to_gemini_contents(messages)
    if not contents:
        contents = [{"role": "user", "parts": [{"text": system_prompt or ""}]}]
        system_prompt = ""

    # Strip the leading "models/" if present
    model_id = model[len("models/"):] if model.startswith("models/") else model

    from google.genai import types
    cfg_kwargs: dict[str, Any] = dict(temperature=temperature)
    if system_prompt:
        cfg_kwargs["system_instruction"] = system_prompt
    max_tokens = extra.pop("max_tokens", None) or extra.pop("max_completion_tokens", None) or extra.pop("max_output_tokens", None)
    if max_tokens:
        cfg_kwargs["max_output_tokens"] = max_tokens
    config = types.GenerateContentConfig(**cfg_kwargs)

    resp = _google().models.generate_content(
        model=model_id,
        contents=contents,
        config=config,
    )
    text = resp.text or ""
    meta = getattr(resp, "usage_metadata", None)
    return ProviderResult(
        text=text,
        input_tokens=getattr(meta, "prompt_token_count", 0) if meta else 0,
        output_tokens=getattr(meta, "candidates_token_count", 0) if meta else 0,
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def call_llm(model: str, messages: list[dict], temperature: float, **extra: Any) -> ProviderResult:
    """Call the right provider for ``model`` and return a uniform result."""
    provider = detect_provider(model)
    if provider == "openai":
        return _openai_call(model, messages, temperature, **extra)
    if provider == "anthropic":
        return _anthropic_call(model, messages, temperature, **extra)
    if provider == "google":
        return _google_call(model, messages, temperature, **extra)
    raise ValueError(f"Unknown provider: {provider}")
