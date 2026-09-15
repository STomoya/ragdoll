"""Answer-correctness metrics: Exact Match, token-level F1."""

from __future__ import annotations

import string
from collections import Counter

from pydantic import BaseModel

_ARTICLES = {'a', 'an', 'the'}


class CorrectnessMetrics(BaseModel):
    """Exact-match and token-F1 scores for one query (or a mean across queries)."""

    exact_match: float
    f1: float


def _normalize(text: str) -> str:
    lowered = text.lower()
    without_punctuation = ''.join(char for char in lowered if char not in string.punctuation)
    tokens = [token for token in without_punctuation.split() if token not in _ARTICLES]
    return ' '.join(tokens)


def _token_f1(prediction_tokens: list[str], gold_tokens: list[str]) -> float:
    if not prediction_tokens or not gold_tokens:
        return float(prediction_tokens == gold_tokens)

    common = Counter(prediction_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_correctness_metrics(answer: str, gold_answers: list[str]) -> CorrectnessMetrics:
    """Score one query's answer against its gold answers, taking the max F1 across gold answers."""
    if not gold_answers:
        return CorrectnessMetrics(exact_match=0.0, f1=0.0)

    normalized_answer = _normalize(answer)
    normalized_golds = [_normalize(gold) for gold in gold_answers]

    exact_match = 1.0 if normalized_answer in normalized_golds else 0.0
    f1 = max(_token_f1(normalized_answer.split(), gold.split()) for gold in normalized_golds)

    return CorrectnessMetrics(exact_match=exact_match, f1=f1)
