"""Tests for IdentityReranker."""

from __future__ import annotations

from ragdoll.core.schema import Query, RetrievedContext
from ragdoll.stages.rerankers.identity import IdentityReranker, IdentityRerankerConfig


def test_returns_contexts_unchanged():
    reranker = IdentityReranker(IdentityRerankerConfig())
    query = Query(query_id='q1', text='q', lang='en')
    contexts = [RetrievedContext(chunk_id='c1', doc_id='d1', text='t', rank=1)]

    result = reranker(query, contexts)

    assert result is contexts
