"""Tests for answer-correctness metrics (Exact Match, token F1)."""

from __future__ import annotations

import pytest

from ragdoll.core.metrics.correctness import compute_correctness_metrics


def test_exact_match_ignores_case_punctuation_and_articles() -> None:
    result = compute_correctness_metrics('The Paris.', gold_answers=['paris'])

    assert result.exact_match == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)


def test_partial_overlap_scores_partial_f1_and_no_exact_match() -> None:
    result = compute_correctness_metrics('The Eiffel Tower is in Paris', gold_answers=['Paris'])

    assert result.exact_match == pytest.approx(0.0)
    assert result.f1 == pytest.approx(0.3333, abs=1e-4)


def test_takes_max_score_across_multiple_gold_answers() -> None:
    result = compute_correctness_metrics('Paris', gold_answers=['London', 'Paris'])

    assert result.exact_match == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)


def test_empty_gold_answers_scores_zero_without_crashing() -> None:
    result = compute_correctness_metrics('anything', gold_answers=[])

    assert result.exact_match == pytest.approx(0.0)
    assert result.f1 == pytest.approx(0.0)


def test_no_token_overlap_scores_zero_f1() -> None:
    result = compute_correctness_metrics('completely unrelated', gold_answers=['Paris'])

    assert result.exact_match == pytest.approx(0.0)
    assert result.f1 == pytest.approx(0.0)
