# RAG Eval — AGENTS.md

Staged, method-agnostic RAG implementation and evaluation harness. The
harness decomposes any RAG system into independently swappable stages
(chunking, query transform, retrieval, reranking, generation), evaluates
combinations of stage implementations against shared benchmarks, and
reports results per exact combination. This file is the standing rulebook
every stage and method implementation must follow — it defines the shared
data model, per-stage contracts, and evaluation rules that make results
comparable across methods. It contains no implementation plan — per-method
and per-stage specs are written separately when that work starts.

## Tech stack

- Language: Python (>=3.11)
- Package manager: uv — `uv add`, never bare `pip install`
- Type checker: ty (Astral), installed as a uv tool — never mypy
- Lint / format: ruff, installed as a uv tool
- Test runner: pytest, with `pytest-mock`'s `mocker` fixture for mocking
- Validation / data model: Pydantic v2 for every shared type below
- License: Apache 2.0

## Data model

All pipelines index the same corpus and answer the same queries, expressed
in a benchmark-agnostic shape. A benchmark adapter converts a benchmark's
native format into this shape once; every stage and every metric then
operates on this shape only — nothing else in the codebase may know a
benchmark's native format.

The canonical definitions are `Document`, `Query`, `Chunk`,
`RetrievedContext`, `TransformedQuery`, `ReasoningStep`, and `RAGResponse`
in `src/ragdoll/core/schema.py` — read that file for exact fields and
defaults; don't duplicate it here. All cross-stage data uses those models
exactly as defined — do not invent parallel ad hoc shapes for convenience.

## Pipeline stages

A pipeline is a composition of up to five stage slots, each a plain
callable with a fixed input/output type. Implementations for a slot are
interchangeable as long as they satisfy that type.

- **Chunking.** `Chunker: list[Document] -> list[Chunk]`. Fixed-size,
  recursive, semantic, layout-aware, or no-op (whole doc = one chunk).
- **Query transformation** (optional; identity by default).
  `QueryTransform: Query -> TransformedQuery`. Identity, HyDE (generate a
  hypothetical answer/document and search with that), query rewriting,
  multi-query expansion, decomposition into sub-questions. This stage never
  sees the retrieval backend — that's what makes it composable across
  retriever families.
- **Retrieval**, two parts because indexing and searching happen at
  different times:
  - `BuildIndex: list[Chunk] -> IndexHandle` (opaque: dense vector index,
    BM25/inverted index, knowledge graph, hybrid store, ...).
  - `Retrieve: (TransformedQuery, IndexHandle) -> list[RetrievedContext]`.
- **Reranking** (optional; identity by default).
  `Rerank: (Query, list[RetrievedContext]) -> list[RetrievedContext]`.
  Cross-encoder rerank, LLM-based rerank, reciprocal-rank fusion, or
  pass-through.
- **Generation.**
  `Generate: (Query, list[RetrievedContext]) -> RAGResponse`. Single-shot
  read, agentic/iterative, graph-grounded synthesis, etc.

## Architecture rules

These apply to every stage implementation and every future method, without
exception:

- A stage implementation touches only its own slot's input/output types
  above. It never imports or reaches into another stage's internals.
- The one allowed exception: a **generator** may re-invoke `Retrieve`
  internally (agentic/multi-hop methods). When it does, every internal
  retrieval call must be appended to `reasoning_trace`, and the final
  `retrieved_contexts` must be the union of everything the answer was
  actually conditioned on. No other stage may do this.
- `BuildIndex` depends only on `(chunker, retriever)`. A retrieval
  implementation must not assume anything about which query-transform,
  reranker, or generator will be used downstream. The harness builds each
  distinct `(chunker, retriever)` index once and reuses it across every
  combination that shares it.
- A query-transform implementation must not assume anything about which
  retrieval backend will consume its `TransformedQuery` — it produces
  `search_texts` and nothing backend-specific.
- If a retriever requires specific chunk properties (e.g. entity-tagged
  chunks for a graph retriever), declare that constraint in the registry.
  Never silently run on incompatible input — the combination must be
  excluded, not degraded.
- Every new stage implementation is registered by name in `core/registry.py`
  and must declare its own config schema (Pydantic model) — no bare dicts
  for hyperparameters.
- Every result is tagged with its full combination id
  (`{chunker, query_transform, retriever, reranker, generator}` plus each
  stage's own config), so results are diffable across combinations.
- All gold-derivation and metric code lives only in `core/metrics/` and
  operates only on `RAGResponse` + gold fields from `Query`. Metric code
  never imports from `stages/`.
- Benchmark adapters (`benchmarks/`) are the only code allowed to know a
  benchmark's native format. Their only job is producing
  `list[Document]` + `list[Query]`.

## Gold derivation

Benchmarks provide gold at the document level (`gold_doc_ids`); since
chunking is itself a variable under test, retrieval metrics need
chunk-level gold. A chunk counts as relevant if its `doc_id` is in
`gold_doc_ids` (document-level benchmarks get coarser-grained credit), with
an optional stricter mode using substring/span overlap between the chunk
text and a known gold passage span, when a benchmark provides one (e.g.
KILT-style provenance).

## Metrics

- **Retrieval quality** (needs gold doc/chunk ids): Recall@k, Precision@k,
  MRR, nDCG@k over `retrieved_contexts`.
- **Answer correctness** (needs `gold_answers`): Exact Match / token-level
  F1 for fixed-answer benchmarks; LLM-judge correctness score for free-form
  ones, with judge model/prompt fixed and versioned so scores stay
  comparable across runs and languages.
- **Faithfulness / groundedness** (RAGAS-style; no gold needed):
  faithfulness (answer claims supported by `retrieved_contexts`), answer
  relevance, context precision/recall. Works identically regardless of
  which stage implementations produced the response.
- **Efficiency**: `latency_ms`, `token_usage`, aggregated p50/p95, plus
  index build time and size per `(chunker, retriever)` pair.
- **Multilingual breakdown**: all of the above sliced by `query.lang`.

## Benchmarks

- **Natural Questions**, via KILT formatting — standard English open-domain
  QA with a fixed Wikipedia corpus, gold answers, and gold supporting
  passages; exercises retrieval and correctness together.
- **MIRACL** — multilingual retrieval benchmark, 18 languages including
  Japanese, part of the BEIR/MTEB family. Real Japanese queries and corpus,
  not machine-translated.
- **MKQA** (optional add-on) — open-domain QA, 26 languages incl. Japanese,
  NQ-derived corpus, for Japanese answer-correctness/faithfulness.

## Repo layout

```
rag-eval/
  src/ragdoll/
    core/
      schema.py            # Document, Query, Chunk, TransformedQuery, RetrievedContext, RAGResponse, ReasoningStep
      registry.py           # stage registries + compatibility constraints
      pipeline.py            # composes stage callables into one run given a combination id
      runner.py               # expands a combination grid, builds/reuses indexes, runs pipelines, computes metrics
      metrics/
        retrieval.py
        correctness.py
        faithfulness.py
        efficiency.py
    benchmarks/
      natural_questions.py
      miracl.py
      mkqa.py
    stages/
      chunkers/
      query_transforms/
      retrievers/
      rerankers/
      generators/
  reports/                  # per-combination metric output (json/csv)
```

## Commands

- Install deps: `uv sync`
- Run Python: `uv run python ...`
- Run tests: `uv run pytest`
- Type check: `uvx ty check`
- Lint: `uvx ruff check .`
- Format: `uvx ruff format .`

All four (tests, ty, ruff check, ruff format --check) must pass before any
task is considered done.

## Coding conventions

- Type-hint everything; `ty check` must be clean, no `# type: ignore`
  without a comment explaining why.
- All cross-stage data uses the Pydantic models above exactly as specified
  — do not invent parallel ad hoc shapes for convenience.
- No stage implementation calls an LLM/embedding provider directly inline;
  route through a shared client wrapper so latency/token accounting
  (`latency_ms`, `token_usage`) is captured uniformly across every method.
- Config for a stage implementation (model name, chunk size, top-k, etc.)
  lives in that implementation's own Pydantic config model, never as
  free-floating constants.
- Never hardcode a benchmark's file layout or language list outside its
  adapter module.

## Branching & workflow

- Never commit directly to `main`. All work happens on a branch; merge to
  `main` only once the branch's tests/lint/typecheck pass.
- Every new stage implementation, benchmark adapter, or metric ships with
  tests before merge — at minimum, a contract-compliance test (does it
  satisfy the declared input/output types and registry constraints) plus
  unit tests for its own logic.
- Registry/compatibility-constraint changes (declaring or removing a
  chunker↔retriever compatibility rule) require a test that the harness
  actually skips the now-incompatible combination.

## Out of scope for this file

- Any specific stage implementation (a chunker, a retriever, HyDE, a
  reranker, a generator) — those get their own spec/plan when work on them
  starts.
- Model/library choices for embeddings, vector stores, graph stores, or
  LLM providers — decided per implementation, not mandated globally.
- Benchmark-adapter implementation details — decided per adapter when that
  work starts.
