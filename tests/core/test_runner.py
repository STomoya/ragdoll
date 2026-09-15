"""Tests for core.runner."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from ragdoll.stages.retrievers.dense import DenseIndex

import ragdoll.stages  # noqa: F401  (registers every stage implementation)
from ragdoll.core.pipeline import StageCombination
from ragdoll.core.runner import build_index, expand_grid, run_combination
from ragdoll.core.schema import Document, Query


def test_expand_grid_produces_cartesian_product() -> None:
    combinations = expand_grid(
        chunkers=['fixed_size'],
        query_transforms=['identity'],
        retrievers=['dense'],
        rerankers=['identity'],
        generators=['single_shot'],
    )

    assert combinations == [
        StageCombination(
            chunker='fixed_size',
            query_transform='identity',
            retriever='dense',
            reranker='identity',
            generator='single_shot',
        ),
    ]


def test_build_index_chunks_and_embeds_documents(mocker: MockerFixture) -> None:
    mock_embed = mocker.patch('ragdoll.stages.retrievers.dense.embed_texts', return_value=([[0.1, 0.2]], 1.0))
    documents = [Document(doc_id='d1', text='short doc')]

    index_handle = build_index('fixed_size', 'dense', documents, chunker_config={'chunk_size': 500, 'overlap': 50})

    dense_index = cast('DenseIndex', index_handle)
    assert len(dense_index.chunks) == 1
    mock_embed.assert_called_once()


def test_run_combination_runs_every_query(mocker: MockerFixture) -> None:
    mocker.patch('ragdoll.stages.retrievers.dense.embed_texts', return_value=([[0.1, 0.2]], 1.0))
    mocker.patch(
        'ragdoll.stages.generators.single_shot.generate_chat',
        return_value=('an answer', 3.0, {'prompt_tokens': 1, 'completion_tokens': 1}),
    )
    documents = [Document(doc_id='d1', text='short doc')]
    index_handle = build_index('fixed_size', 'dense', documents)
    combination = StageCombination(chunker='fixed_size', retriever='dense', generator='single_shot')
    queries = [Query(query_id='q1', text='q1?', lang='en'), Query(query_id='q2', text='q2?', lang='en')]

    responses = run_combination(combination, index_handle, queries)

    assert [r.query_id for r in responses] == ['q1', 'q2']
