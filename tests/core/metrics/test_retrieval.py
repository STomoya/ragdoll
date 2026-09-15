"""Tests for retrieval-quality metrics."""

from __future__ import annotations

import pytest

from ragdoll.core.metrics.retrieval import compute_retrieval_metrics
from ragdoll.core.schema import RetrievedContext


def _context(doc_id: str, rank: int) -> RetrievedContext:
    return RetrievedContext(chunk_id=f'{doc_id}::0', doc_id=doc_id, text='irrelevant', rank=rank)


def test_perfect_retrieval_at_rank_one() -> None:
    contexts = [_context('gold1', rank=1)]

    result = compute_retrieval_metrics(contexts, gold_doc_ids=['gold1'])

    assert result.recall_at_k == pytest.approx(1.0)
    assert result.precision_at_k == pytest.approx(1.0)
    assert result.mrr == pytest.approx(1.0)
    assert result.ndcg_at_k == pytest.approx(1.0)


def test_gold_doc_found_at_second_rank() -> None:
    contexts = [_context('other', rank=1), _context('gold1', rank=2)]

    result = compute_retrieval_metrics(contexts, gold_doc_ids=['gold1'])

    assert result.recall_at_k == pytest.approx(1.0)
    assert result.precision_at_k == pytest.approx(0.5)
    assert result.mrr == pytest.approx(0.5)
    assert result.ndcg_at_k == pytest.approx(0.6309, abs=1e-4)


def test_no_relevant_context_retrieved() -> None:
    contexts = [_context('other', rank=1)]

    result = compute_retrieval_metrics(contexts, gold_doc_ids=['gold1'])

    assert result.recall_at_k == pytest.approx(0.0)
    assert result.precision_at_k == pytest.approx(0.0)
    assert result.mrr == pytest.approx(0.0)
    assert result.ndcg_at_k == pytest.approx(0.0)


def test_empty_gold_doc_ids_scores_zero_without_crashing() -> None:
    contexts = [_context('other', rank=1)]

    result = compute_retrieval_metrics(contexts, gold_doc_ids=[])

    assert result.recall_at_k == pytest.approx(0.0)
    assert result.precision_at_k == pytest.approx(0.0)
    assert result.mrr == pytest.approx(0.0)
    assert result.ndcg_at_k == pytest.approx(0.0)


def test_duplicate_doc_ids_across_chunks_count_once_for_recall() -> None:
    contexts = [_context('gold1', rank=1), _context('gold1', rank=2)]

    result = compute_retrieval_metrics(contexts, gold_doc_ids=['gold1', 'gold2'])

    assert result.recall_at_k == pytest.approx(0.5)  # only gold1 found, gold2 never appears
    assert result.precision_at_k == pytest.approx(1.0)  # both retrieved chunks are from a gold doc
    assert result.mrr == pytest.approx(1.0)  # first context (rank 1) is gold1, which is relevant
    ndcg_expected = 0.6131  # dcg=1.0; idcg≈1.6309; 1.0/1.6309≈0.6131
    assert result.ndcg_at_k == pytest.approx(ndcg_expected, abs=1e-4)
