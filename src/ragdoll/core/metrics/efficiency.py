"""Efficiency metrics: latency percentiles and summed token usage across a run."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

if TYPE_CHECKING:
    from ragdoll.core.schema import RAGResponse


class EfficiencyMetrics(BaseModel):
    """Aggregate latency and token-usage stats across a run's responses."""

    latency_p50_ms: float
    latency_p95_ms: float
    token_usage: dict[str, int]


def compute_efficiency_metrics(responses: list[RAGResponse]) -> EfficiencyMetrics:
    """Aggregate latency percentiles and summed token usage across all responses in a run."""
    if not responses:
        return EfficiencyMetrics(latency_p50_ms=0.0, latency_p95_ms=0.0, token_usage={})

    latencies = [response.latency_ms for response in responses]
    token_usage: dict[str, int] = {}
    for response in responses:
        for key, value in response.token_usage.items():
            token_usage[key] = token_usage.get(key, 0) + value

    return EfficiencyMetrics(
        latency_p50_ms=float(np.percentile(latencies, 50)),
        latency_p95_ms=float(np.percentile(latencies, 95)),
        token_usage=token_usage,
    )
