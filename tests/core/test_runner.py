"""Tests for core.runner."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_mock import MockerFixture

    from ragdoll.stages.retrievers.dense import DenseIndex

import ragdoll.core.runner
import ragdoll.stages
from ragdoll.core import registry
from ragdoll.core.pipeline import StageCombination
from ragdoll.core.runner import IncompatibleStagesError, build_index, expand_grid, run_combination
from ragdoll.core.schema import Document, Query


@pytest.fixture
def incompatible_fixed_size_dense() -> Iterator[None]:
    """Temporarily declare fixed_size/dense incompatible, then undeclare it."""
    registry.declare_incompatible('fixed_size', 'dense', 'test reason')
    try:
        yield
    finally:
        registry._incompatible.pop(('fixed_size', 'dense'), None)


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


def test_run_combination_builds_stages_once_for_all_queries(mocker: MockerFixture) -> None:
    mocker.patch('ragdoll.stages.retrievers.dense.embed_texts', return_value=([[0.1, 0.2]], 1.0))
    mocker.patch(
        'ragdoll.stages.generators.single_shot.generate_chat',
        return_value=('an answer', 3.0, {'prompt_tokens': 1, 'completion_tokens': 1}),
    )
    build_stages_spy = mocker.patch(
        'ragdoll.core.runner.build_pipeline_stages',
        wraps=ragdoll.core.runner.build_pipeline_stages,
    )
    documents = [Document(doc_id='d1', text='short doc')]
    index_handle = build_index('fixed_size', 'dense', documents)
    combination = StageCombination(chunker='fixed_size', retriever='dense', generator='single_shot')
    queries = [Query(query_id='q1', text='q1?', lang='en'), Query(query_id='q2', text='q2?', lang='en')]

    run_combination(combination, index_handle, queries)

    # stages are instantiated once per combination and reused across queries,
    # not re-instantiated (with a fresh registry lookup + config validation) per query.
    build_stages_spy.assert_called_once_with(combination)


@pytest.mark.usefixtures('incompatible_fixed_size_dense')
def test_expand_grid_excludes_incompatible_pair() -> None:
    # 'other_chunker' isn't a registered implementation; expand_grid only
    # filters by name compatibility here, it never resolves the registry.
    combinations = expand_grid(
        chunkers=['fixed_size', 'other_chunker'],
        query_transforms=['identity'],
        retrievers=['dense'],
        rerankers=['identity'],
        generators=['single_shot'],
    )

    assert combinations == [
        StageCombination(
            chunker='other_chunker',
            query_transform='identity',
            retriever='dense',
            reranker='identity',
            generator='single_shot',
        ),
    ]


@pytest.mark.usefixtures('incompatible_fixed_size_dense')
def test_build_index_raises_for_incompatible_pair() -> None:
    documents = [Document(doc_id='d1', text='short doc')]

    with pytest.raises(IncompatibleStagesError):
        build_index('fixed_size', 'dense', documents)
