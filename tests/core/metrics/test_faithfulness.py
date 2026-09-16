"""Tests for the LLM-judged faithfulness metric."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

from ragdoll.core.metrics.faithfulness import score_faithfulness
from ragdoll.core.schema import Query, RAGResponse, RetrievedContext

_MOCK_LATENCY_MS = 12.0
_MOCK_TOKEN_USAGE = {'prompt_tokens': 5, 'completion_tokens': 1}


def _query_and_response(answer: str) -> tuple[Query, RAGResponse]:
    query = Query(query_id='q1', text='what is rag?', lang='en')
    contexts = [RetrievedContext(chunk_id='c1', doc_id='d1', text='RAG combines retrieval and generation.', rank=1)]
    response = RAGResponse(query_id='q1', answer=answer, retrieved_contexts=contexts, latency_ms=3.0, token_usage={})
    return query, response


def test_parses_well_formed_score(mocker: MockerFixture) -> None:
    mocker.patch(
        'ragdoll.core.metrics.faithfulness.generate_chat',
        return_value=('0.75', _MOCK_LATENCY_MS, _MOCK_TOKEN_USAGE),
    )
    query, response = _query_and_response('RAG combines retrieval and generation.')

    score = score_faithfulness(query, response)

    assert score == pytest.approx(0.75)


def test_clamps_out_of_range_score(mocker: MockerFixture) -> None:
    mocker.patch(
        'ragdoll.core.metrics.faithfulness.generate_chat',
        return_value=('1.5', _MOCK_LATENCY_MS, _MOCK_TOKEN_USAGE),
    )
    query, response = _query_and_response('RAG combines retrieval and generation.')

    score = score_faithfulness(query, response)

    assert score == pytest.approx(1.0)


def test_unparseable_judge_response_scores_zero(mocker: MockerFixture) -> None:
    mocker.patch(
        'ragdoll.core.metrics.faithfulness.generate_chat',
        return_value=('I cannot determine this.', _MOCK_LATENCY_MS, _MOCK_TOKEN_USAGE),
    )
    query, response = _query_and_response('anything')

    score = score_faithfulness(query, response)

    assert score == pytest.approx(0.0)


def test_uses_given_judge_model_instead_of_default(mocker: MockerFixture) -> None:
    mock_generate_chat = mocker.patch(
        'ragdoll.core.metrics.faithfulness.generate_chat',
        return_value=('1.0', _MOCK_LATENCY_MS, _MOCK_TOKEN_USAGE),
    )
    query, response = _query_and_response('anything')

    score_faithfulness(query, response, judge_model='a-different-model')

    assert mock_generate_chat.call_args.args[1] == 'a-different-model'
