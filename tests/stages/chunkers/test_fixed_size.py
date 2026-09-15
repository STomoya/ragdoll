"""Tests for FixedSizeChunker."""

from __future__ import annotations

import pytest

from ragdoll.core.schema import Chunk, Document
from ragdoll.stages.chunkers.fixed_size import FixedSizeChunker, FixedSizeChunkerConfig


def test_splits_long_document_into_overlapping_chunks():
    chunker = FixedSizeChunker(FixedSizeChunkerConfig(chunk_size=10, overlap=2))
    documents = [Document(doc_id='d1', text='0123456789abcdefghij')]

    chunks = chunker(documents)

    assert all(isinstance(c, Chunk) for c in chunks)
    assert [c.text for c in chunks] == ['0123456789', '89abcdefgh', 'ghij']
    assert [c.chunk_id for c in chunks] == ['d1::0', 'd1::1', 'd1::2']
    assert [c.position for c in chunks] == [0, 1, 2]
    assert all(c.doc_id == 'd1' for c in chunks)


def test_short_document_produces_single_chunk():
    chunker = FixedSizeChunker(FixedSizeChunkerConfig(chunk_size=500, overlap=50))
    documents = [Document(doc_id='d1', text='short text')]

    chunks = chunker(documents)

    assert len(chunks) == 1
    assert chunks[0].text == 'short text'


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError, match='overlap must be smaller than chunk_size'):
        FixedSizeChunkerConfig(chunk_size=10, overlap=10)
