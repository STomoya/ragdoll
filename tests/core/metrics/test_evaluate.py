"""Integration test: evaluate_combination composes all four metric modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

from ragdoll.core.metrics.evaluate import evaluate_combination
from ragdoll.core.pipeline import StageCombination
from ragdoll.core.schema import Query, RAGResponse, RetrievedContext

_MOCK_LATENCY_MS = 5.0
_MOCK_TOKEN_USAGE = {'prompt_tokens': 1, 'completion_tokens': 1}


def test_evaluate_combination_scores_every_query_and_tags_the_combination(mocker: MockerFixture) -> None:
    mocker.patch(
        'ragdoll.core.metrics.faithfulness.generate_chat',
        return_value=('1.0', _MOCK_LATENCY_MS, _MOCK_TOKEN_USAGE),
    )
    combination = StageCombination(chunker='fixed_size', retriever='dense', generator='single_shot')
    queries = [
        Query(query_id='q1', text='q1?', lang='en', gold_answers=['answer one'], gold_doc_ids=['d1']),
        Query(query_id='q2', text='q2?', lang='en', gold_answers=['answer two'], gold_doc_ids=['d2']),
    ]
    responses = [
        RAGResponse(
            query_id='q1',
            answer='answer one',
            retrieved_contexts=[RetrievedContext(chunk_id='c1', doc_id='d1', text='ctx', rank=1)],
            latency_ms=100.0,
            token_usage={'prompt_tokens': 10, 'completion_tokens': 2},
        ),
        RAGResponse(
            query_id='q2',
            answer='wrong answer',
            retrieved_contexts=[RetrievedContext(chunk_id='c2', doc_id='other', text='ctx', rank=1)],
            latency_ms=200.0,
            token_usage={'prompt_tokens': 20, 'completion_tokens': 4},
        ),
    ]

    result = evaluate_combination(combination, responses, queries)

    assert result.combination == combination
    assert result.n_queries == pytest.approx(2)
    # q1 is a perfect retrieval+answer, q2 misses both -> means land at 0.5
    assert result.retrieval.recall_at_k == pytest.approx(0.5)
    assert result.correctness.exact_match == pytest.approx(0.5)
    assert result.efficiency.token_usage == {'prompt_tokens': 30, 'completion_tokens': 6}
    assert result.faithfulness.faithfulness == pytest.approx(1.0)
