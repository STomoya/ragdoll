"""Shared client wrapper for embedding and LLM calls; the only module that imports third-party AI providers."""

from __future__ import annotations

import time
from functools import cache
from typing import Any, cast

from openai import OpenAI
from sentence_transformers import SentenceTransformer


@cache
def _load_embedding_model(model_name: str, device: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device=device)


def embed_texts(texts: list[str], model_name: str, device: str = 'cpu') -> tuple[Any, float]:
    """Embed texts with a local sentence-transformers model; returns (embeddings, latency_ms)."""
    model = _load_embedding_model(model_name, device)
    start = time.perf_counter()
    embeddings = model.encode(texts)
    latency_ms = (time.perf_counter() - start) * 1000
    return embeddings, latency_ms


@cache
def _get_openai_client() -> OpenAI:
    return OpenAI()


def generate_chat(
    messages: list[dict[str, str]],
    model_name: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, float, dict[str, int]]:
    """Call an OpenAI-compatible chat endpoint; returns (answer_text, latency_ms, token_usage)."""
    client = _get_openai_client()
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model_name,
        messages=cast('Any', messages),
        max_tokens=max_tokens,
        temperature=temperature,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    content = response.choices[0].message.content if response.choices else None
    token_usage = {
        'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
        'completion_tokens': response.usage.completion_tokens if response.usage else 0,
    }
    return content or '', latency_ms, token_usage
