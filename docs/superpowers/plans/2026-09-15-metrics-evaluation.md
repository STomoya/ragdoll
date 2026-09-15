# Automatic Metrics Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the harness a function that automatically scores a stage combination's `RAGResponse`s against a benchmark's gold fields (retrieval quality, answer correctness, efficiency, faithfulness), instead of requiring manual inspection.

**Architecture:** Four independent, pure per-metric modules under `src/ragdoll/core/metrics/` (`retrieval.py`, `correctness.py`, `efficiency.py`, `faithfulness.py`), each exporting one Pydantic result model and one compute function. A fifth module, `evaluate.py`, matches `RAGResponse`s to `Query`s by `query_id`, calls each metric function per query, averages, and returns one `EvaluationResult` tagged with the run's `StageCombination`.

**Tech Stack:** Python 3.11+, Pydantic v2, numpy (already a dependency), pytest + pytest-mock, ruff, ty.

**Spec:** `docs/superpowers/specs/2026-09-15-metrics-evaluation-design.md`

## Global Constraints

- Package manager: `uv` — never bare `pip install` (no new dependencies needed for this plan; numpy is already installed).
- Type-hint everything; `uvx ty check` must be clean, no `# type: ignore` without a comment explaining why.
- Format/lint: `uvx ruff format .` and `uvx ruff check .` must both pass clean — ruff's `PL` (pylint) rules are enabled, so no magic-value literals in comparisons inside tests (use a named constant, per `tests/stages/generators/test_single_shot.py`'s `_MOCK_LATENCY_MS` pattern).
- Docstrings required on every public class/function (ruff `D`, Google convention) — one line is enough unless non-obvious behavior needs explaining.
- All new Pydantic models/functions live only in `src/ragdoll/core/metrics/` — this code must never import from `ragdoll.stages`.
- Never commit directly to `main` — this work continues on the current branch (`feature/basic-rag-pipeline`).
- Every new module ships with tests before the task is considered done: `uv run pytest` must pass.

---

## File Structure

```
src/ragdoll/core/metrics/retrieval.py      # RetrievalMetrics, compute_retrieval_metrics (replaces docstring-only stub)
src/ragdoll/core/metrics/correctness.py    # CorrectnessMetrics, compute_correctness_metrics (replaces docstring-only stub)
src/ragdoll/core/metrics/efficiency.py     # EfficiencyMetrics, compute_efficiency_metrics (replaces docstring-only stub)
src/ragdoll/core/metrics/faithfulness.py   # FaithfulnessMetrics, score_faithfulness (replaces docstring-only stub)
src/ragdoll/core/metrics/evaluate.py       # EvaluationResult, evaluate_combination (new file)
tests/core/metrics/__init__.py             # new empty test package
tests/core/metrics/test_retrieval.py
tests/core/metrics/test_correctness.py
tests/core/metrics/test_efficiency.py
tests/core/metrics/test_faithfulness.py
tests/core/metrics/test_evaluate.py
```

---

### Task 1: Retrieval quality metrics

**Files:**
- Create: `tests/core/metrics/__init__.py` (empty)
- Modify: `src/ragdoll/core/metrics/retrieval.py` (currently just a docstring)
- Test: `tests/core/metrics/test_retrieval.py`

**Interfaces:**
- Consumes: `ragdoll.core.schema.RetrievedContext` (fields: `chunk_id`, `doc_id`, `text`, `score`, `rank`) — already defined, no changes.
- Produces: `RetrievalMetrics(BaseModel)` with fields `recall_at_k: float`, `precision_at_k: float`, `mrr: float`, `ndcg_at_k: float`; `compute_retrieval_metrics(retrieved_contexts: list[RetrievedContext], gold_doc_ids: list[str], k: int | None = None) -> RetrievalMetrics`. Later tasks (`evaluate.py`) import both names from `ragdoll.core.metrics.retrieval`.

- [ ] **Step 1: Create the empty test package marker**

```bash
touch tests/core/metrics/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/core/metrics/test_retrieval.py`:

```python
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
    assert result.precision_at_k == pytest.approx(0.5)  # 1 distinct relevant doc / 2 retrieved
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/core/metrics/test_retrieval.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_retrieval_metrics'`

- [ ] **Step 4: Implement retrieval metrics**

Replace the contents of `src/ragdoll/core/metrics/retrieval.py`:

```python
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

    denominator = min(k, len(retrieved_contexts))
    precision_at_k = len(relevant_doc_ids) / denominator if denominator > 0 else 0.0

    mrr = 0.0
    for rank, context in enumerate(top_k, start=1):
        if context.doc_id in gold_ids:
            mrr = 1.0 / rank
            break

    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, context in enumerate(top_k, start=1) if context.doc_id in gold_ids
    )
    ideal_hits = min(len(gold_ids), len(top_k))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg_at_k = dcg / idcg if idcg > 0 else 0.0

    return RetrievalMetrics(recall_at_k=recall_at_k, precision_at_k=precision_at_k, mrr=mrr, ndcg_at_k=ndcg_at_k)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/core/metrics/test_retrieval.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Type-check and lint**

Run: `uvx ty check && uvx ruff check src/ragdoll/core/metrics/retrieval.py tests/core/metrics/test_retrieval.py && uvx ruff format src/ragdoll/core/metrics/retrieval.py tests/core/metrics/test_retrieval.py`
Expected: all clean

- [ ] **Step 7: Commit**

```bash
git add src/ragdoll/core/metrics/retrieval.py tests/core/metrics/test_retrieval.py tests/core/metrics/__init__.py
git commit -m "Add retrieval-quality metrics (Recall@k, Precision@k, MRR, nDCG@k)"
```

---

### Task 2: Answer-correctness metrics

**Files:**
- Modify: `src/ragdoll/core/metrics/correctness.py` (currently just a docstring)
- Test: `tests/core/metrics/test_correctness.py`

**Interfaces:**
- Consumes: nothing beyond plain `str`/`list[str]`.
- Produces: `CorrectnessMetrics(BaseModel)` with fields `exact_match: float`, `f1: float`; `compute_correctness_metrics(answer: str, gold_answers: list[str]) -> CorrectnessMetrics`. Later tasks (`evaluate.py`) import both names from `ragdoll.core.metrics.correctness`.

- [ ] **Step 1: Write the failing tests**

Create `tests/core/metrics/test_correctness.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/metrics/test_correctness.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_correctness_metrics'`

- [ ] **Step 3: Implement correctness metrics**

Replace the contents of `src/ragdoll/core/metrics/correctness.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/metrics/test_correctness.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Type-check and lint**

Run: `uvx ty check && uvx ruff check src/ragdoll/core/metrics/correctness.py tests/core/metrics/test_correctness.py && uvx ruff format src/ragdoll/core/metrics/correctness.py tests/core/metrics/test_correctness.py`
Expected: all clean

- [ ] **Step 6: Commit**

```bash
git add src/ragdoll/core/metrics/correctness.py tests/core/metrics/test_correctness.py
git commit -m "Add answer-correctness metrics (Exact Match, token F1)"
```

---

### Task 3: Efficiency metrics

**Files:**
- Modify: `src/ragdoll/core/metrics/efficiency.py` (currently just a docstring)
- Test: `tests/core/metrics/test_efficiency.py`

**Interfaces:**
- Consumes: `ragdoll.core.schema.RAGResponse` (fields used: `latency_ms: float`, `token_usage: dict[str, int]`) — already defined, no changes.
- Produces: `EfficiencyMetrics(BaseModel)` with fields `latency_p50_ms: float`, `latency_p95_ms: float`, `token_usage: dict[str, int]`; `compute_efficiency_metrics(responses: list[RAGResponse]) -> EfficiencyMetrics`. Later tasks (`evaluate.py`) import both names from `ragdoll.core.metrics.efficiency`.

- [ ] **Step 1: Write the failing tests**

Create `tests/core/metrics/test_efficiency.py`:

```python
"""Tests for efficiency metrics (latency percentiles, summed token usage)."""

from __future__ import annotations

import pytest

from ragdoll.core.metrics.efficiency import compute_efficiency_metrics
from ragdoll.core.schema import RAGResponse


def _response(query_id: str, latency_ms: float, token_usage: dict[str, int]) -> RAGResponse:
    return RAGResponse(query_id=query_id, answer='a', retrieved_contexts=[], latency_ms=latency_ms, token_usage=token_usage)


def test_aggregates_latency_percentiles_and_sums_token_usage() -> None:
    responses = [
        _response('q1', latency_ms=100.0, token_usage={'prompt_tokens': 10, 'completion_tokens': 2}),
        _response('q2', latency_ms=200.0, token_usage={'prompt_tokens': 20, 'completion_tokens': 4}),
    ]

    result = compute_efficiency_metrics(responses)

    assert result.latency_p50_ms == pytest.approx(150.0)
    assert result.latency_p95_ms == pytest.approx(195.0)
    assert result.token_usage == {'prompt_tokens': 30, 'completion_tokens': 6}


def test_empty_responses_scores_zero_without_crashing() -> None:
    result = compute_efficiency_metrics([])

    assert result.latency_p50_ms == pytest.approx(0.0)
    assert result.latency_p95_ms == pytest.approx(0.0)
    assert result.token_usage == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/metrics/test_efficiency.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_efficiency_metrics'`

- [ ] **Step 3: Implement efficiency metrics**

Replace the contents of `src/ragdoll/core/metrics/efficiency.py`:

```python
"""Efficiency metrics: latency percentiles and summed token usage across a run."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

if TYPE_CHECKING:
    from ragdoll.core.schema import RAGResponse


class EfficiencyMetrics(BaseModel):
    """Aggregate latency and token-usage stats across a run's responses."""

    latency_p50_ms: float
    latency_p95_ms: float
    token_usage: dict[str, int]


def compute_efficiency_metrics(responses: list[RAGResponse]) -> EfficiencyMetrics:
    """Aggregate latency percentiles and summed token usage across all responses in a run."""
    if not responses:
        return EfficiencyMetrics(latency_p50_ms=0.0, latency_p95_ms=0.0, token_usage={})

    latencies = [response.latency_ms for response in responses]
    token_usage: dict[str, int] = {}
    for response in responses:
        for key, value in response.token_usage.items():
            token_usage[key] = token_usage.get(key, 0) + value

    return EfficiencyMetrics(
        latency_p50_ms=float(np.percentile(latencies, 50)),
        latency_p95_ms=float(np.percentile(latencies, 95)),
        token_usage=token_usage,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/metrics/test_efficiency.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Type-check and lint**

Run: `uvx ty check && uvx ruff check src/ragdoll/core/metrics/efficiency.py tests/core/metrics/test_efficiency.py && uvx ruff format src/ragdoll/core/metrics/efficiency.py tests/core/metrics/test_efficiency.py`
Expected: all clean

- [ ] **Step 6: Commit**

```bash
git add src/ragdoll/core/metrics/efficiency.py tests/core/metrics/test_efficiency.py
git commit -m "Add efficiency metrics (latency p50/p95, summed token usage)"
```

---

### Task 4: Faithfulness metric (LLM judge)

**Files:**
- Modify: `src/ragdoll/core/metrics/faithfulness.py` (currently just a docstring)
- Test: `tests/core/metrics/test_faithfulness.py`

**Interfaces:**
- Consumes: `ragdoll.core.clients.generate_chat(messages, model_name, max_tokens, temperature) -> tuple[str, float, dict[str, int]]` (already defined, unchanged); `ragdoll.core.schema.Query` (field used: `text`), `ragdoll.core.schema.RAGResponse` (fields used: `answer`, `retrieved_contexts`) — already defined, no changes.
- Produces: `FaithfulnessMetrics(BaseModel)` with field `faithfulness: float`; `score_faithfulness(query: Query, response: RAGResponse) -> float`. Later tasks (`evaluate.py`) import both names from `ragdoll.core.metrics.faithfulness`.

- [ ] **Step 1: Write the failing tests**

Create `tests/core/metrics/test_faithfulness.py`:

```python
"""Tests for the LLM-judged faithfulness metric."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

from ragdoll.core.metrics.faithfulness import score_faithfulness
from ragdoll.core.schema import Query, RAGResponse, RetrievedContext

_MOCK_LATENCY_MS = 12.0
_MOCK_TOKEN_USAGE = {'prompt_tokens': 5, 'completion_tokens': 1}


def _query_and_response(answer: str) -> tuple[Query, RAGResponse]:
    query = Query(query_id='q1', text='what is rag?', lang='en')
    contexts = [RetrievedContext(chunk_id='c1', doc_id='d1', text='RAG combines retrieval and generation.', rank=1)]
    response = RAGResponse(query_id='q1', answer=answer, retrieved_contexts=contexts, latency_ms=3.0, token_usage={})
    return query, response


def test_parses_well_formed_score(mocker: MockerFixture) -> None:
    mocker.patch(
        'ragdoll.core.metrics.faithfulness.generate_chat',
        return_value=('0.75', _MOCK_LATENCY_MS, _MOCK_TOKEN_USAGE),
    )
    query, response = _query_and_response('RAG combines retrieval and generation.')

    score = score_faithfulness(query, response)

    assert score == pytest.approx(0.75)


def test_clamps_out_of_range_score(mocker: MockerFixture) -> None:
    mocker.patch(
        'ragdoll.core.metrics.faithfulness.generate_chat',
        return_value=('1.5', _MOCK_LATENCY_MS, _MOCK_TOKEN_USAGE),
    )
    query, response = _query_and_response('RAG combines retrieval and generation.')

    score = score_faithfulness(query, response)

    assert score == pytest.approx(1.0)


def test_unparseable_judge_response_scores_zero(mocker: MockerFixture) -> None:
    mocker.patch(
        'ragdoll.core.metrics.faithfulness.generate_chat',
        return_value=('I cannot determine this.', _MOCK_LATENCY_MS, _MOCK_TOKEN_USAGE),
    )
    query, response = _query_and_response('anything')

    score = score_faithfulness(query, response)

    assert score == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/metrics/test_faithfulness.py -v`
Expected: FAIL with `ImportError: cannot import name 'score_faithfulness'`

- [ ] **Step 3: Implement the faithfulness judge**

Replace the contents of `src/ragdoll/core/metrics/faithfulness.py`:

```python
"""Faithfulness metric: an LLM judge scores whether an answer's claims are grounded in its contexts."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ragdoll.core.clients import generate_chat

if TYPE_CHECKING:
    from ragdoll.core.schema import Query, RAGResponse

_JUDGE_MODEL = 'gpt-4o-mini'
_JUDGE_PROMPT_VERSION = '2026-09-15-v1'
_JUDGE_MAX_TOKENS = 16
_JUDGE_TEMPERATURE = 0.0
_SCORE_PATTERN = re.compile(r'(\d*\.?\d+)')


class FaithfulnessMetrics(BaseModel):
    """Mean LLM-judged faithfulness score across a run's queries."""

    faithfulness: float


def score_faithfulness(query: Query, response: RAGResponse) -> float:
    """Ask an LLM judge what fraction of the answer's claims are supported by its retrieved contexts."""
    context_block = '\n\n'.join(
        f'[{i}] {context.text}' for i, context in enumerate(response.retrieved_contexts, start=1)
    )
    prompt = (
        'You are grading whether an answer is faithful to its source contexts.\n\n'
        f'Question: {query.text}\n\n'
        f'Contexts:\n{context_block}\n\n'
        f'Answer: {response.answer}\n\n'
        'What fraction of the claims in the answer are directly supported by the '
        'contexts above? Respond with only a single number between 0.0 and 1.0.'
    )
    # ponytail: the judge call's own latency/token cost isn't tracked here; if
    # judge overhead needs accounting, extend this to return (score, latency_ms,
    # token_usage) and fold it into EfficiencyMetrics in evaluate.py.
    judged_text, _latency_ms, _token_usage = generate_chat(
        [{'role': 'user', 'content': prompt}],
        _JUDGE_MODEL,
        _JUDGE_MAX_TOKENS,
        _JUDGE_TEMPERATURE,
    )
    match = _SCORE_PATTERN.search(judged_text)
    if match is None:
        return 0.0
    return min(max(float(match.group(1)), 0.0), 1.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/metrics/test_faithfulness.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Type-check and lint**

Run: `uvx ty check && uvx ruff check src/ragdoll/core/metrics/faithfulness.py tests/core/metrics/test_faithfulness.py && uvx ruff format src/ragdoll/core/metrics/faithfulness.py tests/core/metrics/test_faithfulness.py`
Expected: all clean

- [ ] **Step 6: Commit**

```bash
git add src/ragdoll/core/metrics/faithfulness.py tests/core/metrics/test_faithfulness.py
git commit -m "Add LLM-judged faithfulness metric"
```

---

### Task 5: Composition — `evaluate_combination`

**Files:**
- Create: `src/ragdoll/core/metrics/evaluate.py`
- Test: `tests/core/metrics/test_evaluate.py`

**Interfaces:**
- Consumes: `RetrievalMetrics`/`compute_retrieval_metrics` from Task 1, `CorrectnessMetrics`/`compute_correctness_metrics` from Task 2, `EfficiencyMetrics`/`compute_efficiency_metrics` from Task 3, `FaithfulnessMetrics`/`score_faithfulness` from Task 4, and `ragdoll.core.pipeline.StageCombination` (already defined, unchanged).
- Produces: `EvaluationResult(BaseModel)` with fields `combination: StageCombination`, `n_queries: int`, `retrieval: RetrievalMetrics`, `correctness: CorrectnessMetrics`, `efficiency: EfficiencyMetrics`, `faithfulness: FaithfulnessMetrics`; `evaluate_combination(combination: StageCombination, responses: list[RAGResponse], queries: list[Query]) -> EvaluationResult`. This is the public entry point other code (scripts, notebooks) calls.

- [ ] **Step 1: Write the failing test**

Create `tests/core/metrics/test_evaluate.py`:

```python
"""Integration test: evaluate_combination composes all four metric modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

from ragdoll.core.metrics.evaluate import evaluate_combination
from ragdoll.core.pipeline import StageCombination
from ragdoll.core.schema import Query, RAGResponse, RetrievedContext

_MOCK_LATENCY_MS = 5.0
_MOCK_TOKEN_USAGE = {'prompt_tokens': 1, 'completion_tokens': 1}


def test_evaluate_combination_scores_every_query_and_tags_the_combination(mocker: MockerFixture) -> None:
    mocker.patch(
        'ragdoll.core.metrics.faithfulness.generate_chat',
        return_value=('1.0', _MOCK_LATENCY_MS, _MOCK_TOKEN_USAGE),
    )
    combination = StageCombination(chunker='fixed_size', retriever='dense', generator='single_shot')
    queries = [
        Query(query_id='q1', text='q1?', lang='en', gold_answers=['answer one'], gold_doc_ids=['d1']),
        Query(query_id='q2', text='q2?', lang='en', gold_answers=['answer two'], gold_doc_ids=['d2']),
    ]
    responses = [
        RAGResponse(
            query_id='q1',
            answer='answer one',
            retrieved_contexts=[RetrievedContext(chunk_id='c1', doc_id='d1', text='ctx', rank=1)],
            latency_ms=100.0,
            token_usage={'prompt_tokens': 10, 'completion_tokens': 2},
        ),
        RAGResponse(
            query_id='q2',
            answer='wrong answer',
            retrieved_contexts=[RetrievedContext(chunk_id='c2', doc_id='other', text='ctx', rank=1)],
            latency_ms=200.0,
            token_usage={'prompt_tokens': 20, 'completion_tokens': 4},
        ),
    ]

    result = evaluate_combination(combination, responses, queries)

    assert result.combination == combination
    assert result.n_queries == pytest.approx(2)
    # q1 is a perfect retrieval+answer, q2 misses both -> means land at 0.5
    assert result.retrieval.recall_at_k == pytest.approx(0.5)
    assert result.correctness.exact_match == pytest.approx(0.5)
    assert result.efficiency.token_usage == {'prompt_tokens': 30, 'completion_tokens': 6}
    assert result.faithfulness.faithfulness == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/metrics/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ragdoll.core.metrics.evaluate'`

- [ ] **Step 3: Implement the composition function**

Create `src/ragdoll/core/metrics/evaluate.py`:

```python
"""Composes the four metric modules into one EvaluationResult for a stage combination's run."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from ragdoll.core.metrics.correctness import CorrectnessMetrics, compute_correctness_metrics
from ragdoll.core.metrics.efficiency import EfficiencyMetrics, compute_efficiency_metrics
from ragdoll.core.metrics.faithfulness import FaithfulnessMetrics, score_faithfulness
from ragdoll.core.metrics.retrieval import RetrievalMetrics, compute_retrieval_metrics
from ragdoll.core.pipeline import StageCombination

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
) -> EvaluationResult:
    """Score a combination's responses against their queries' gold fields, averaged across queries."""
    queries_by_id = {query.query_id: query for query in queries}

    retrieval_scores: list[RetrievalMetrics] = []
    correctness_scores: list[CorrectnessMetrics] = []
    faithfulness_scores: list[float] = []

    for response in responses:
        query = queries_by_id[response.query_id]
        retrieval_scores.append(compute_retrieval_metrics(response.retrieved_contexts, query.gold_doc_ids))
        correctness_scores.append(compute_correctness_metrics(response.answer, query.gold_answers))
        faithfulness_scores.append(score_faithfulness(query, response))

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
        faithfulness=FaithfulnessMetrics(faithfulness=_mean(faithfulness_scores)),
    )
```

Note: `StageCombination` must be a real (non-`TYPE_CHECKING`) import here — it's used as an actual Pydantic field type on `EvaluationResult`, and Pydantic v2 resolves field annotations at class-definition time, so it must be importable at runtime, not just for static type checking.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/core/metrics/test_evaluate.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Run the full suite and quality gates**

Run: `uv run pytest -q && uvx ty check && uvx ruff check . && uvx ruff format --check .`
Expected: all tests pass, all checks clean

- [ ] **Step 6: Commit**

```bash
git add src/ragdoll/core/metrics/evaluate.py tests/core/metrics/test_evaluate.py
git commit -m "Add evaluate_combination composing retrieval/correctness/efficiency/faithfulness metrics"
```

---

## Post-plan verification

After Task 5, run the full quality-gate sequence from `CLAUDE.md` one more time to confirm nothing regressed:

```bash
uv run pytest -q
uvx ty check
uvx ruff check .
uvx ruff format --check .
```

All four must pass before this work is considered done.
