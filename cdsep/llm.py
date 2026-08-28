"""Provider-agnostic LLM client with retry, logging, cost tracking, and caching.

Dispatches to OpenAI, Anthropic, or Google Gemini based on the model name; see
:mod:`cdsep.providers` for the per-provider adapters.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from cdsep.logging_utils import ExperimentLogger
from cdsep.providers import call_llm, detect_provider


# Pricing (USD per 1M tokens). Input / output. Used only for cost-tracking.
PRICING_PER_1M = {
    # OpenAI
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    # Anthropic
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-5": {"input": 15.00, "output": 75.00},
    # Google
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
}

_DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".llm_cache")


class LLMCache:
    """Disk-backed cache keyed on (model, temperature, messages).

    Only caches deterministic calls (temperature <= 0). Non-deterministic
    calls (temperature > 0, e.g. optimizer) are never cached.
    """

    def __init__(self, cache_dir: str = _DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _make_key(model: str, temperature: float, messages: list[dict]) -> str:
        # NOTE: we intentionally do NOT include provider in the key because
        # the model name uniquely identifies the provider in our dispatch.
        blob = json.dumps(
            {"model": model, "temperature": temperature, "messages": messages},
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, model: str, temperature: float, messages: list[dict]) -> str | None:
        if temperature > 0:
            return None
        key = self._make_key(model, temperature, messages)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self.hits += 1
            with open(path) as f:
                return json.load(f)["response"]
        self.misses += 1
        return None

    def put(self, model: str, temperature: float, messages: list[dict], response: str) -> None:
        if temperature > 0:
            return
        key = self._make_key(model, temperature, messages)
        path = self.cache_dir / f"{key}.json"
        with open(path, "w") as f:
            json.dump({"model": model, "temperature": temperature, "response": response}, f)


_shared_cache = LLMCache()


class LLMClient:
    """Provider-agnostic chat client with caching, retry, and logging.

    The provider (OpenAI / Anthropic / Google) is inferred from the model
    name. The interface is the same regardless of provider.
    """

    def __init__(
        self,
        model: str = "gpt-5.4-nano",
        temperature: float = 1.0,
        max_retries: int = 3,
        logger: ExperimentLogger | None = None,
        cache: LLMCache | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.logger = logger
        self.cache = cache if cache is not None else _shared_cache
        self.provider = detect_provider(model)

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self.cache_hits = 0

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Send a chat completion request and return the response text."""
        temp = temperature if temperature is not None else self.temperature

        cached = self.cache.get(self.model, temp, messages)
        if cached is not None:
            self.cache_hits += 1
            self.total_calls += 1
            if self.logger:
                self.logger.log_llm_call(
                    model=self.model, messages=messages,
                    response_text=cached, usage=None, latency_ms=0,
                    cache_hit=True,
                )
            return cached

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.time()
                result = call_llm(self.model, messages, temperature=temp, **kwargs)
                latency_ms = (time.time() - t0) * 1000

                text = result.text
                self.total_input_tokens += result.input_tokens
                self.total_output_tokens += result.output_tokens
                self.total_calls += 1
                self.cache.put(self.model, temp, messages, text)

                if self.logger:
                    self.logger.log_llm_call(
                        model=self.model, messages=messages,
                        response_text=text,
                        usage={"prompt_tokens": result.input_tokens,
                               "completion_tokens": result.output_tokens},
                        latency_ms=round(latency_ms, 1),
                        cache_hit=False,
                    )
                return text

            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                time.sleep(wait)

        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_error}")

    @property
    def estimated_cost(self) -> float:
        pricing = PRICING_PER_1M.get(self.model, {"input": 1.0, "output": 3.0})
        return (
            self.total_input_tokens * pricing["input"] / 1_000_000
            + self.total_output_tokens * pricing["output"] / 1_000_000
        )
