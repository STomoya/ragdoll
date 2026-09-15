"""Integration test: full pipeline through registered real stages, only network/model calls mocked."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ragdoll.stages  # noqa: F401  (registers every stage implementation)
from ragdoll.core.pipeline import StageCombination, run_pipeline
from ragdoll.core.registry import retrievers
from ragdoll.core.schema import Chunk, Query

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def test_run_pipeline_end_to_end(mocker: MockerFixture) -> None:
    mocker.patch('ragdoll.stages.retrievers.dense.embed_texts', return_value=([[0.1, 0.2]], 1.0))
    mocker.patch(
        'ragdoll.stages.generators.single_shot.generate_chat',
        return_value=('an answer', 3.0, {'prompt_tokens': 1, 'completion_tokens': 1}),
    )

    retriever_entry = retrievers.get('dense')
    retriever = retriever_entry.cls(retriever_entry.config_model())
    chunks = [Chunk(chunk_id='c1', doc_id='d1', text='some context', position=0)]
    index_handle = retriever.build_index(chunks)

    combination = StageCombination(chunker='fixed_size', retriever='dense', generator='single_shot')
    query = Query(query_id='q1', text='what is rag?', lang='en')

    response = run_pipeline(combination, index_handle, query)

    assert response.query_id == 'q1'
    assert response.answer == 'an answer'
    assert len(response.retrieved_contexts) == 1
