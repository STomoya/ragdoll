"""Retrieval-quality metrics: Recall@k, Precision@k, MRR, nDCG@k."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from ragdoll.core.schema import RetrievedContext


class RetrievalMetrics(BaseModel):
    """Retrieval-quality scores for one query (or a mean across queries)."""

    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float


def compute_retrieval_metrics(
    retrieved_contexts: list[RetrievedContext],
    gold_doc_ids: list[str],
    k: int | None = None,
) -> RetrievalMetrics:
    """Score one query's retrieved contexts against its gold document ids."""
    if not gold_doc_ids:
        return RetrievalMetrics(recall_at_k=0.0, precision_at_k=0.0, mrr=0.0, ndcg_at_k=0.0)

    k = len(retrieved_contexts) if k is None else k
    top_k = retrieved_contexts[:k]
    gold_ids = set(gold_doc_ids)

    relevant_doc_ids = {context.doc_id for context in top_k if context.doc_id in gold_ids}
    recall_at_k = len(relevant_doc_ids) / len(gold_ids)

    relevant_chunks = sum(1 for context in top_k if context.doc_id in gold_ids)
    precision_at_k = relevant_chunks / k if k > 0 else 0.0

    mrr = 0.0
    for rank, context in enumerate(top_k, start=1):
        if context.doc_id in gold_ids:
            mrr = 1.0 / rank
            break

    seen_doc_ids: set[str] = set()
    dcg = 0.0
    for rank, context in enumerate(top_k, start=1):
        if context.doc_id in gold_ids and context.doc_id not in seen_doc_ids:
            dcg += 1.0 / math.log2(rank + 1)
            seen_doc_ids.add(context.doc_id)
    ideal_hits = min(len(gold_ids), len(top_k))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg_at_k = dcg / idcg if idcg > 0 else 0.0

    return RetrievalMetrics(recall_at_k=recall_at_k, precision_at_k=precision_at_k, mrr=mrr, ndcg_at_k=ndcg_at_k)
