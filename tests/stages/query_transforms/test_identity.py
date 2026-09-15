"""Tests for IdentityQueryTransform."""

from __future__ import annotations

from ragdoll.core.schema import Query, TransformedQuery
from ragdoll.stages.query_transforms.identity import IdentityQueryTransform, IdentityQueryTransformConfig


def test_returns_query_text_as_single_search_text():
    transform = IdentityQueryTransform(IdentityQueryTransformConfig())
    query = Query(query_id='q1', text='what is rag?', lang='en')

    result = transform(query)

    assert isinstance(result, TransformedQuery)
    assert result.query_id == 'q1'
    assert result.search_texts == ['what is rag?']
