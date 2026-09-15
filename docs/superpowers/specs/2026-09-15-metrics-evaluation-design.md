# Automatic metrics evaluation — design

Status: approved by user (chat), 2026-09-15.

## Goal

Give the harness a way to automatically score a combination's results
against a benchmark's gold fields, instead of requiring manual inspection of
`RAGResponse` objects. This spec covers four of CLAUDE.md's five metric
categories: retrieval quality, answer correctness (EM/F1 only), efficiency,
and faithfulness (the single faithfulness score only, not the full RAGAS
triad). Out of scope for this pass: answer relevance, context
precision/recall, LLM-judge correctness for free-form benchmarks, and
multilingual breakdown (all current benchmark data is English-only via NQ —
this becomes relevant once MIRACL/MKQA are wired up). A results-persistence
mechanism (writing to `reports/`) is also out of scope — `evaluate_combination`
returns a Pydantic model; the caller decides how to save it.

## Data model

All new models live in `core/metrics/`, not `schema.py` — they're evaluation
output, not cross-stage data.

`core/metrics/retrieval.py`:

```python
class RetrievalMetrics(BaseModel):
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
```

`core/metrics/correctness.py`:

```python
class CorrectnessMetrics(BaseModel):
    exact_match: float  # 0.0 or 1.0 per query, mean once aggregated
    f1: float
```

`core/metrics/efficiency.py`:

```python
class EfficiencyMetrics(BaseModel):
    latency_p50_ms: float
    latency_p95_ms: float
    token_usage: dict[str, int]  # summed per key across all responses
```

`core/metrics/faithfulness.py`:

```python
class FaithfulnessMetrics(BaseModel):
    faithfulness: float  # mean judge score across queries, 0.0-1.0
```

`core/metrics/evaluate.py`:

```python
class EvaluationResult(BaseModel):
    combination: StageCombination
    n_queries: int
    retrieval: RetrievalMetrics
    correctness: CorrectnessMetrics
    efficiency: EfficiencyMetrics
    faithfulness: FaithfulnessMetrics
```

## Gold derivation

Retrieval relevance is `RetrievedContext.doc_id in query.gold_doc_ids` —
doc-level, per CLAUDE.md's gold-derivation default. No chunk lookup is
needed since `RetrievedContext` already carries `doc_id`. `k` defaults to
`len(retrieved_contexts)` (whatever the retriever actually returned); callers
may pass an explicit `k` to slice further. If `gold_doc_ids` is empty, all
four retrieval metrics are defined as `0.0` (rather than raising a
division-by-zero) — this is a query data issue, not a pipeline failure, and
should show up as a visibly bad score rather than crash the eval run. Same
`0.0`-on-empty rule applies to correctness when `gold_answers` is empty.

## Retrieval metrics (`core/metrics/retrieval.py`)

```python
def compute_retrieval_metrics(
    retrieved_contexts: list[RetrievedContext],
    gold_doc_ids: list[str],
    k: int | None = None,
) -> RetrievalMetrics:
```

Relevance is checked over `retrieved_contexts[:k]`; "relevant docs retrieved"
means the count of *distinct* `gold_doc_ids` appearing among those `k`
contexts' `doc_id`s (dedup by doc, since a chunker can produce multiple
chunks — and so multiple retrieved contexts — for the same gold document).

- `recall_at_k` = `|relevant docs retrieved| / |gold_doc_ids|`
- `precision_at_k` = `|relevant docs retrieved| / min(k, len(retrieved_contexts))`
- `mrr` = `1 / rank` of the first relevant context (1-indexed), `0.0` if none
- `ndcg_at_k` = binary-relevance DCG@k / ideal DCG@k, standard `log2(rank+1)`
  discount

## Correctness metrics (`core/metrics/correctness.py`)

```python
def compute_correctness_metrics(answer: str, gold_answers: list[str]) -> CorrectnessMetrics:
```

SQuAD-style normalization before comparing: lowercase, strip punctuation,
drop English articles (`a`/`an`/`the`), collapse whitespace. `exact_match` is
`1.0` if the normalized answer equals any normalized gold answer, else
`0.0`. `f1` is token-multiset overlap F1 against each gold answer,
independently, taking the max across `gold_answers`.

## Efficiency metrics (`core/metrics/efficiency.py`)

```python
def compute_efficiency_metrics(responses: list[RAGResponse]) -> EfficiencyMetrics:
```

Aggregates across the whole list of responses (not per-query — percentiles
need multiple samples): `latency_p50_ms`/`latency_p95_ms` via
`numpy.percentile` over each response's `latency_ms`; `token_usage` sums
each key present across all responses' `token_usage` dicts.

## Faithfulness (`core/metrics/faithfulness.py`)

```python
def score_faithfulness(query: Query, response: RAGResponse) -> float:
```

One `generate_chat` call per response, through the existing `core/clients.py`
wrapper (no new provider code). Fixed, versioned prompt and judge model as
module constants:

```python
_JUDGE_MODEL = "gpt-4o-mini"
_JUDGE_PROMPT_VERSION = "2026-09-15-v1"
```

The prompt gives the judge the query, the answer, and the numbered
`retrieved_contexts` text, and asks for a single float 0.0–1.0 rating of
"what fraction of the answer's claims are supported by the given contexts."
The response is parsed for a float in `[0.0, 1.0]`; if parsing fails, the
score is `0.0` (a judge that can't produce a usable score counts as not
having verified faithfulness, not as an error that aborts the eval run).

## Composition (`core/metrics/evaluate.py`)

```python
def evaluate_combination(
    combination: StageCombination,
    responses: list[RAGResponse],
    queries: list[Query],
) -> EvaluationResult:
```

Matches each `response` to its `query` by `query_id` (a dict keyed by
`query_id` built from `queries`), computes retrieval/correctness/faithfulness
per query, then averages each metric field across queries (plain arithmetic
mean — this is the standard way Recall@k/Precision@k/MRR/nDCG@k/EM/F1 are
reported over a query set). `compute_efficiency_metrics` runs once over the
full `responses` list, since it aggregates directly. `n_queries =
len(responses)`.

## File layout additions

```
src/ragdoll/core/metrics/retrieval.py      # RetrievalMetrics, compute_retrieval_metrics (currently a docstring-only stub)
src/ragdoll/core/metrics/correctness.py    # CorrectnessMetrics, compute_correctness_metrics (currently a docstring-only stub)
src/ragdoll/core/metrics/efficiency.py     # EfficiencyMetrics, compute_efficiency_metrics (currently a docstring-only stub)
src/ragdoll/core/metrics/faithfulness.py   # FaithfulnessMetrics, score_faithfulness (currently a docstring-only stub)
src/ragdoll/core/metrics/evaluate.py       # EvaluationResult, evaluate_combination (new file)
tests/core/metrics/test_retrieval.py
tests/core/metrics/test_correctness.py
tests/core/metrics/test_efficiency.py
tests/core/metrics/test_faithfulness.py
tests/core/metrics/test_evaluate.py
```

## Testing

- Unit tests per metric module: known inputs with hand-computed expected
  Recall/Precision/MRR/nDCG, EM/F1 (including the empty-gold `0.0` edge
  case), and efficiency aggregation (percentiles/summed token usage) over
  fixed lists of responses.
- `faithfulness.py` tests mock `generate_chat` (via `pytest-mock`, matching
  the existing pattern in `tests/stages/generators/test_single_shot.py`) —
  no real LLM call in the test suite — and cover both a well-formed score
  response and an unparseable one (asserting the `0.0` fallback).
- `evaluate.py` gets an integration-style test: synthetic `Query`/
  `RAGResponse` fixtures (2-3 queries) with `generate_chat` mocked, asserting
  the returned `EvaluationResult` is tagged with the given `combination` and
  that `n_queries` and each metrics group's shape are correct.
