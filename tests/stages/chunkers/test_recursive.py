"""Tests for RecursiveTextSplitter."""

from __future__ import annotations

import pytest

from ragdoll.core.schema import Chunk, Document
from ragdoll.stages.chunkers.recursive import RecursiveTextSplitter, RecursiveTextSplitterConfig


def test_splits_on_paragraph_boundaries_when_they_fit():
    chunker = RecursiveTextSplitter(RecursiveTextSplitterConfig(chunk_size=20, overlap=0))
    documents = [Document(doc_id='d1', text='first paragraph\n\nsecond paragraph\n\nthird')]

    chunks = chunker(documents)

    assert all(isinstance(c, Chunk) for c in chunks)
    assert [c.text for c in chunks] == ['first paragraph', 'second paragraph', 'third']
    assert [c.chunk_id for c in chunks] == ['d1::0', 'd1::1', 'd1::2']
    assert [c.position for c in chunks] == [0, 1, 2]
    assert all(c.doc_id == 'd1' for c in chunks)


def test_chunks_are_exact_substrings_of_the_source_document():
    text = 'one two three four five six seven eight nine ten'
    chunker = RecursiveTextSplitter(RecursiveTextSplitterConfig(chunk_size=15, overlap=3))
    documents = [Document(doc_id='d1', text=text)]

    chunks = chunker(documents)

    assert all(chunk.text in text for chunk in chunks)


def test_packs_small_pieces_together_up_to_chunk_size():
    chunk_size = 12
    chunker = RecursiveTextSplitter(RecursiveTextSplitterConfig(chunk_size=chunk_size, overlap=0))
    documents = [Document(doc_id='d1', text='a\nb\nc\nd\ne\nf\ng\nh')]

    chunks = chunker(documents)

    assert all(len(c.text) <= chunk_size for c in chunks)
    assert ''.join(c.text for c in chunks).replace('\n', '') == 'abcdefgh'


def test_falls_back_to_char_splitting_for_an_unsplittable_run():
    chunker = RecursiveTextSplitter(RecursiveTextSplitterConfig(chunk_size=5, overlap=0))
    documents = [Document(doc_id='d1', text='abcdefghij')]

    chunks = chunker(documents)

    assert [c.text for c in chunks] == ['abcde', 'fghij']


def test_short_document_produces_single_chunk():
    chunker = RecursiveTextSplitter(RecursiveTextSplitterConfig(chunk_size=500, overlap=50))
    documents = [Document(doc_id='d1', text='short text')]

    chunks = chunker(documents)

    assert len(chunks) == 1
    assert chunks[0].text == 'short text'


def test_empty_document_produces_no_chunks():
    chunker = RecursiveTextSplitter(RecursiveTextSplitterConfig(chunk_size=10, overlap=0))
    documents = [Document(doc_id='d1', text='')]

    assert chunker(documents) == []


def test_consecutive_separators_do_not_produce_empty_chunks():
    chunker = RecursiveTextSplitter(RecursiveTextSplitterConfig(chunk_size=6, overlap=0))
    documents = [Document(doc_id='d1', text='first\n\n\n\nsecond')]

    chunks = chunker(documents)

    assert [c.text for c in chunks] == ['first', 'second']


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError, match='overlap must be smaller than chunk_size'):
        RecursiveTextSplitterConfig(chunk_size=10, overlap=10)
