"""Composes the four metric modules into one EvaluationResult for a stage combination's run."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from ragdoll.core.metrics.correctness import CorrectnessMetrics, compute_correctness_metrics
from ragdoll.core.metrics.efficiency import EfficiencyMetrics, compute_efficiency_metrics
from ragdoll.core.metrics.faithfulness import DEFAULT_JUDGE_MODEL, FaithfulnessMetrics, score_faithfulness
from ragdoll.core.metrics.retrieval import RetrievalMetrics, compute_retrieval_metrics
from ragdoll.core.pipeline import StageCombination  # noqa: TC001 (used as a Pydantic field type; needed at runtime)

if TYPE_CHECKING:
    from ragdoll.core.schema import Query, RAGResponse


class EvaluationResult(BaseModel):
    """Metrics for one stage combination's run over a query set, tagged with that combination."""

    combination: StageCombination
    n_queries: int
    retrieval: RetrievalMetrics
    correctness: CorrectnessMetrics
    efficiency: EfficiencyMetrics
    faithfulness: FaithfulnessMetrics


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_combination(
    combination: StageCombination,
    responses: list[RAGResponse],
    queries: list[Query],
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> EvaluationResult:
    """Score a combination's responses against their queries' gold fields, averaged across queries.

    Pass judge_model explicitly when it would otherwise match the generator
    under test, to avoid self-preference bias in the faithfulness score.
    """
    queries_by_id = {query.query_id: query for query in queries}

    retrieval_scores: list[RetrievalMetrics] = []
    correctness_scores: list[CorrectnessMetrics] = []
    faithfulness_scores: list[float] = []

    for response in responses:
        query = queries_by_id[response.query_id]
        retrieval_scores.append(compute_retrieval_metrics(response.retrieved_contexts, query.gold_doc_ids))
        correctness_scores.append(compute_correctness_metrics(response.answer, query.gold_answers))
        faithfulness_scores.append(score_faithfulness(query, response, judge_model=judge_model))

    return EvaluationResult(
        combination=combination,
        n_queries=len(responses),
        retrieval=RetrievalMetrics(
            recall_at_k=_mean([m.recall_at_k for m in retrieval_scores]),
            precision_at_k=_mean([m.precision_at_k for m in retrieval_scores]),
            mrr=_mean([m.mrr for m in retrieval_scores]),
            ndcg_at_k=_mean([m.ndcg_at_k for m in retrieval_scores]),
        ),
        correctness=CorrectnessMetrics(
            exact_match=_mean([m.exact_match for m in correctness_scores]),
            f1=_mean([m.f1 for m in correctness_scores]),
        ),
        efficiency=compute_efficiency_metrics(responses),
        faithfulness=FaithfulnessMetrics(faithfulness=_mean(faithfulness_scores), judge_model=judge_model),
    )
