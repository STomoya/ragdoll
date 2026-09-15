"""Faithfulness metric: an LLM judge scores whether an answer's claims are grounded in its contexts."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ragdoll.core.clients import generate_chat

if TYPE_CHECKING:
    from ragdoll.core.schema import Query, RAGResponse

# Deliberately different from any generator's default model: a model tends to
# score its own outputs more favorably (self-preference bias), which would
# bias faithfulness comparisons across combinations. Pass judge_model
# explicitly if a generator under test happens to use this same model.
DEFAULT_JUDGE_MODEL = 'gpt-4o'
_JUDGE_PROMPT_VERSION = '2026-09-15-v2'
# Reasoning models spend part of max_tokens on hidden reasoning tokens before
# any visible output, so this needs headroom beyond just the score digit or
# the judge call returns an empty completion.
_JUDGE_MAX_TOKENS = 8192
_JUDGE_TEMPERATURE = 0.0
_SCORE_PATTERN = re.compile(r'(\d*\.?\d+)')


class FaithfulnessMetrics(BaseModel):
    """Mean LLM-judged faithfulness score across a run's queries."""

    faithfulness: float
    judge_model: str


def score_faithfulness(query: Query, response: RAGResponse, judge_model: str = DEFAULT_JUDGE_MODEL) -> float:
    """Ask an LLM judge what fraction of the answer's claims are supported by its retrieved contexts."""
    context_block = '\n\n'.join(
        f'[{i}] {context.text}' for i, context in enumerate(response.retrieved_contexts, start=1)
    )
    prompt = (
        'You are grading whether an answer is faithful to its source contexts.\n\n'
        f'Question: {query.text}\n\n'
        f'Contexts:\n{context_block}\n\n'
        f'Answer: {response.answer}\n\n'
        'What fraction of the claims in the answer are directly supported by the '
        'contexts above? Respond with only a single number between 0.0 and 1.0.'
    )
    # ponytail: the judge call's own latency/token cost isn't tracked here; if
    # judge overhead needs accounting, extend this to return (score, latency_ms,
    # token_usage) and fold it into EfficiencyMetrics in evaluate.py.
    judged_text, _latency_ms, _token_usage = generate_chat(
        [{'role': 'user', 'content': prompt}],
        judge_model,
        _JUDGE_MAX_TOKENS,
        _JUDGE_TEMPERATURE,
    )
    match = _SCORE_PATTERN.search(judged_text)
    if match is None:
        return 0.0
    return min(max(float(match.group(1)), 0.0), 1.0)
