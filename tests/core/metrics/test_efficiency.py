"""Tests for efficiency metrics (latency percentiles, summed token usage)."""

from __future__ import annotations

import pytest

from ragdoll.core.metrics.efficiency import compute_efficiency_metrics
from ragdoll.core.schema import RAGResponse


def _response(query_id: str, latency_ms: float, token_usage: dict[str, int]) -> RAGResponse:
    return RAGResponse(
        query_id=query_id, answer='a', retrieved_contexts=[], latency_ms=latency_ms, token_usage=token_usage
    )


def test_aggregates_latency_percentiles_and_sums_token_usage() -> None:
    responses = [
        _response('q1', latency_ms=100.0, token_usage={'prompt_tokens': 10, 'completion_tokens': 2}),
        _response('q2', latency_ms=200.0, token_usage={'prompt_tokens': 20, 'completion_tokens': 4}),
    ]

    result = compute_efficiency_metrics(responses)

    assert result.latency_p50_ms == pytest.approx(150.0)
    assert result.latency_p95_ms == pytest.approx(195.0)
    assert result.token_usage == {'prompt_tokens': 30, 'completion_tokens': 6}


def test_empty_responses_scores_zero_without_crashing() -> None:
    result = compute_efficiency_metrics([])

    assert result.latency_p50_ms == pytest.approx(0.0)
    assert result.latency_p95_ms == pytest.approx(0.0)
    assert result.token_usage == {}
