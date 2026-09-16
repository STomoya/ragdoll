"""Tests for SingleShotGenerator."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

from ragdoll.core.schema import Query, RAGResponse, RetrievedContext
from ragdoll.stages.generators.single_shot import SingleShotGenerator, SingleShotGeneratorConfig

_MOCK_LATENCY_MS = 42.0
_MOCK_PROMPT_TOKENS = 10
_MOCK_COMPLETION_TOKENS = 3


def test_builds_prompt_and_wraps_response(mocker: MockerFixture) -> None:
    mock_generate = mocker.patch(
        'ragdoll.stages.generators.single_shot.generate_chat',
        return_value=(
            'the answer',
            _MOCK_LATENCY_MS,
            {'prompt_tokens': _MOCK_PROMPT_TOKENS, 'completion_tokens': _MOCK_COMPLETION_TOKENS},
        ),
    )
    query = Query(query_id='q1', text='what is rag?', lang='en')
    contexts = [
        RetrievedContext(chunk_id='c1', doc_id='d1', text='RAG stands for retrieval-augmented generation.', rank=1),
    ]
    generator = SingleShotGenerator(SingleShotGeneratorConfig())

    response = generator(query, contexts)

    assert isinstance(response, RAGResponse)
    assert response.query_id == 'q1'
    assert response.answer == 'the answer'
    assert response.retrieved_contexts == contexts
    assert response.latency_ms == _MOCK_LATENCY_MS
    assert response.token_usage == {'prompt_tokens': _MOCK_PROMPT_TOKENS, 'completion_tokens': _MOCK_COMPLETION_TOKENS}

    messages = mock_generate.call_args.args[0]
    assert 'what is rag?' in messages[0]['content']
    assert 'RAG stands for retrieval-augmented generation.' in messages[0]['content']
