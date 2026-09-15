"""Tests for DenseRetriever."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ragdoll.core.schema import Chunk, TransformedQuery
from ragdoll.stages.retrievers.dense import DenseIndex, DenseRetriever, DenseRetrieverConfig

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def test_build_index_embeds_chunk_texts(mocker: MockerFixture) -> None:
    mock_embed = mocker.patch(
        'ragdoll.stages.retrievers.dense.embed_texts',
        return_value=(np.array([[1.0, 0.0], [0.0, 1.0]]), 5.0),
    )
    chunks = [
        Chunk(chunk_id='c1', doc_id='d1', text='alpha', position=0),
        Chunk(chunk_id='c2', doc_id='d1', text='beta', position=1),
    ]
    retriever = DenseRetriever(DenseRetrieverConfig())

    index_handle = retriever.build_index(chunks)

    mock_embed.assert_called_once_with(['alpha', 'beta'], DenseRetrieverConfig().embedding_model)
    assert isinstance(index_handle, DenseIndex)
    assert index_handle.chunks == chunks


def test_retrieve_ranks_by_cosine_similarity(mocker: MockerFixture) -> None:
    mocker.patch(
        'ragdoll.stages.retrievers.dense.embed_texts',
        side_effect=[
            (np.array([[1.0, 0.0], [0.0, 1.0]]), 5.0),
            (np.array([[0.9, 0.1]]), 2.0),
        ],
    )
    chunks = [
        Chunk(chunk_id='c1', doc_id='d1', text='alpha', position=0),
        Chunk(chunk_id='c2', doc_id='d1', text='beta', position=1),
    ]
    retriever = DenseRetriever(DenseRetrieverConfig(top_k=2))
    index_handle = retriever.build_index(chunks)

    results = retriever.retrieve(TransformedQuery(query_id='q1', search_texts=['query text']), index_handle)

    assert [r.chunk_id for r in results] == ['c1', 'c2']
    assert [r.rank for r in results] == [1, 2]
    assert results[0].score is not None
    assert results[1].score is not None
    assert results[0].score > results[1].score
