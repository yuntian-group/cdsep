"""Tests for the multi-provider LLM dispatch."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cdsep.providers import (
    ProviderResult,
    _split_system,
    _to_gemini_contents,
    detect_provider,
)


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def test_detect_openai():
    assert detect_provider("gpt-5.4-nano") == "openai"
    assert detect_provider("gpt-4o-mini") == "openai"
    assert detect_provider("o3-mini") == "openai"


def test_detect_anthropic():
    assert detect_provider("claude-haiku-4-5") == "anthropic"
    assert detect_provider("claude-sonnet-4-6") == "anthropic"


def test_detect_google():
    assert detect_provider("gemini-2.5-flash") == "google"
    assert detect_provider("models/gemini-2.5-pro") == "google"


def test_detect_unknown():
    with pytest.raises(ValueError):
        detect_provider("llama-3.3")


# ---------------------------------------------------------------------------
# Anthropic message split
# ---------------------------------------------------------------------------

def test_split_system_extracts_system():
    msgs = [
        {"role": "system", "content": "You are X."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    sys, others = _split_system(msgs)
    assert sys == "You are X."
    assert others == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]


def test_split_system_concatenates_multiple_system():
    msgs = [
        {"role": "system", "content": "S1"},
        {"role": "user", "content": "U"},
        {"role": "system", "content": "S2"},
    ]
    sys, others = _split_system(msgs)
    assert sys == "S1\n\nS2"
    assert others == [{"role": "user", "content": "U"}]


# ---------------------------------------------------------------------------
# Gemini content conversion
# ---------------------------------------------------------------------------

def test_to_gemini_contents_roles():
    msgs = [
        {"role": "system", "content": "You are X."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Now do Y"},
    ]
    sys, contents = _to_gemini_contents(msgs)
    assert sys == "You are X."
    assert len(contents) == 3
    assert contents[0]["role"] == "user"
    assert contents[1]["role"] == "model"  # assistant -> model
    assert contents[2]["role"] == "user"
    assert contents[0]["parts"][0]["text"] == "Hi"


def test_to_gemini_contents_empty():
    sys, contents = _to_gemini_contents([{"role": "system", "content": "S"}])
    assert sys == "S"
    assert contents == []


# ---------------------------------------------------------------------------
# LLMClient with mocked dispatcher
# ---------------------------------------------------------------------------

def test_llm_client_dispatches_to_provider(tmp_path):
    from cdsep.llm import LLMClient, LLMCache

    cache = LLMCache(cache_dir=str(tmp_path / "cache_a"))

    with patch("cdsep.llm.call_llm") as mock_call:
        mock_call.return_value = ProviderResult(text="hello", input_tokens=10, output_tokens=2)
        client = LLMClient(model="claude-haiku-4-5", temperature=1.0, cache=cache)
        out = client.chat([{"role": "user", "content": "hi"}])
        assert out == "hello"
        assert client.total_calls == 1
        assert client.total_input_tokens == 10
        assert client.total_output_tokens == 2
        assert client.provider == "anthropic"
        mock_call.assert_called_once()


def test_llm_client_caches_deterministic_calls(tmp_path):
    from cdsep.llm import LLMClient, LLMCache

    cache = LLMCache(cache_dir=str(tmp_path / "cache_b"))

    with patch("cdsep.llm.call_llm") as mock_call:
        mock_call.return_value = ProviderResult(text="cached", input_tokens=5, output_tokens=1)
        client = LLMClient(model="gpt-5.4-nano", temperature=0, cache=cache)
        out1 = client.chat([{"role": "user", "content": "x-cache-test-uniq"}])
        out2 = client.chat([{"role": "user", "content": "x-cache-test-uniq"}])
        assert out1 == out2 == "cached"
        # Underlying provider called only once; second call hit the cache
        assert mock_call.call_count == 1
        assert client.cache_hits == 1
