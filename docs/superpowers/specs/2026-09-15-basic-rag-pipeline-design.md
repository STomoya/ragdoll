# Basic RAG pipeline — design

Status: approved by user (chat), 2026-09-15.

## Goal

Turn the scaffolded stubs (schema, registry, empty `pipeline.py`/`runner.py`,
empty stage packages) into one real, working combination: fixed-size
chunking → identity query transform → dense (local embeddings) retrieval →
identity reranking → single-shot LLM generation, evaluated against a small,
real slice of the Natural Questions / KILT benchmark.

This spec only covers *this* combination and the wiring needed to run it. It
does not add new metrics, new benchmarks, or additional stage
implementations — those stay out of scope per AGENTS.md.

## New dependencies (`uv add`)

- `sentence-transformers` — local embedding model for the dense retriever.
- `openai` — OpenAI-compatible chat client for generation.
- `datasets` — Hugging Face `datasets` library, to load KILT/NQ.
- `numpy` — used directly by the dense retriever for cosine similarity.

## Stage calling convention (new — not fixed elsewhere)

Every registered stage class is constructed with its own Pydantic config
instance, then called with only the data, so the constructed instance's call
signature matches the arrow type documented in AGENTS.md exactly:

- Chunker: `chunker = ChunkerCls(config)`; `chunker(documents) -> list[Chunk]`
- QueryTransform: `qt = QTCls(config)`; `qt(query) -> TransformedQuery`
- Retriever: `r = RetrieverCls(config)`; `r.build_index(chunks) -> IndexHandle`
  and `r.retrieve(transformed_query, index_handle) -> list[RetrievedContext]`
  (two methods, matching retrieval's two arrows)
- Reranker: `rr = RerankerCls(config)`; `rr(query, contexts) -> list[RetrievedContext]`
- Generator: `g = GeneratorCls(config)`; `g(query, contexts) -> RAGResponse`

These shapes are expressed as `typing.Protocol`s in `core/protocols.py`
(structural — no base class required of implementations), so `ty` can check
conformance and `pipeline.py`/`runner.py` can dispatch generically without
knowing concrete stage classes.

## Shared client wrapper: `core/clients.py`

Two functions (not classes — no per-call state needed beyond a cached
client/model instance):

- `embed_texts(texts: list[str], model_name: str) -> tuple[np.ndarray, float]`
  — wraps `sentence_transformers.SentenceTransformer`, returns embeddings and
  latency_ms. The `SentenceTransformer` instance is cached per `model_name`
  (`functools.lru_cache` on a loader function) so repeated calls in one
  process don't reload the model.
- `generate_chat(messages: list[dict[str, str]], model_name: str, max_tokens: int, temperature: float) -> tuple[str, float, dict[str, int]]`
  — wraps `openai.OpenAI().chat.completions.create`; the client reads
  `OPENAI_API_KEY` / `OPENAI_BASE_URL` from the environment. Returns the
  answer text, latency_ms, and `{"prompt_tokens": ..., "completion_tokens": ...}`
  from the response's `usage`.

No stage imports `sentence_transformers` or `openai` directly — only
`core/clients.py` does, per AGENTS.md's client-wrapper rule.

## Stage implementations

- `stages/chunkers/fixed_size.py` — `FixedSizeChunker`.
  Config `FixedSizeChunkerConfig(chunk_size: int = 500, overlap: int = 50)`
  (characters). Splits each `Document.text` into overlapping windows;
  `chunk_id = f"{doc_id}::{position}"`, `position` = 0-based window index.
  Registered as `"fixed_size"`.

- `stages/query_transforms/identity.py` — `IdentityQueryTransform`.
  Config: empty `IdentityQueryTransformConfig`. `search_texts=[query.text]`.
  Registered as `"identity"`.

- `stages/retrievers/dense.py` — `DenseRetriever`.
  Config `DenseRetrieverConfig(embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", top_k: int = 5)`.
  `build_index`: embeds all chunk texts via `embed_texts`, returns a
  (module-private, not registered) `DenseIndex` dataclass holding
  `chunks: list[Chunk]` and `embeddings: np.ndarray` — this is exactly the
  opaque `IndexHandle` from AGENTS.md; nothing outside this module inspects
  its structure. `retrieve`: embeds `search_texts` (joined with a space if
  more than one), brute-force cosine similarity against the index array
  (numpy, no faiss — the small scoped corpus doesn't need it), returns the
  top `top_k` as `RetrievedContext` with `rank` starting at 1.
  Registered as `"dense"`.

- `stages/rerankers/identity.py` — `IdentityReranker`.
  Config: empty `IdentityRerankerConfig`. Returns `contexts` unchanged.
  Registered as `"identity"`.

- `stages/generators/single_shot.py` — `SingleShotGenerator`.
  Config `SingleShotGeneratorConfig(model_name: str = "gpt-4o-mini", max_tokens: int = 512, temperature: float = 0.0)`.
  Builds a prompt from `query.text` plus the numbered context texts, calls
  `generate_chat`, wraps the result into a `RAGResponse` (`retrieved_contexts`
  = the contexts it was given, `reasoning_trace=[]`, no internal `Retrieve`
  calls — single-shot only). Registered as `"single_shot"`.

## Pipeline wiring

`core/pipeline.py::run_pipeline(combination, index_handle, query)`: looks up
each of query_transform/retriever/reranker/generator by name in its registry,
instantiates with `entry.config_model(**combination.<slot>_config)`, and runs
transform → retrieve → rerank → generate in sequence, returning the
`RAGResponse`.

`core/runner.py`:
- `expand_grid(chunkers, query_transforms, retrievers, rerankers, generators)`
  — cartesian product of the five name lists into `StageCombination`s with
  default (empty-dict) configs. (Config sweeps per combination are out of
  scope here — AGENTS.md doesn't require them for this pass.)
- `build_index(chunker, retriever, documents, chunker_config=None, retriever_config=None)`
  — **signature change from the current stub**: adds the two config dicts
  (defaulting to `{}`), since building a real index needs the chunker's
  `chunk_size`/`overlap` and the retriever's `embedding_model`. Looks up and
  instantiates the chunker, produces chunks, looks up and instantiates the
  retriever, calls `retriever.build_index(chunks)`.
- `run_combination(combination, index_handle, queries)` — loops queries,
  calling `pipeline.run_pipeline` for each, returns the list of `RAGResponse`.

## Benchmark adapter: `benchmarks/natural_questions.py`

Confirmed via the Hugging Face Hub (dataset structure inspected directly,
not guessed):

- `facebook/kilt_tasks`, config `"nq"` — small (≤14MB/split), has Hub parquet
  export, loads without `trust_remote_code`. Row shape: `id`, `input`
  (question text), `output` (list of `{answer, provenance: list of
  {wikipedia_id, title, ...}}`).
- `facebook/kilt_wikipedia`, config `"2019-08-01"`, split `"full"` — the
  Wikipedia knowledge source. This is a *script-based* dataset
  (`trust_remote_code=True` required) that downloads one ~30GB+ JSON-lines
  file; row shape: `wikipedia_id`, `wikipedia_title`, `text.paragraph` (list
  of paragraph strings). Streaming mode (`streaming=True`) reads it lazily
  over HTTP without a full download.

Adapter function:

```python
def load_natural_questions(
    split: str = "validation",
    n_queries: int = 5,
    n_distractors: int = 20,
) -> tuple[list[Document], list[Query]]:
```

1. Load `facebook/kilt_tasks` (`"nq"`, `split`) fully (small), take the first
   `n_queries` rows. Build `Query`s: `query_id=id`, `text=input`, `lang="en"`,
   `gold_answers` = non-empty `output[*].answer`, `gold_doc_ids` = the unique
   `wikipedia_id`s across `output[*].provenance`.
2. Stream `facebook/kilt_wikipedia` (`"2019-08-01"`, `split="full"`,
   `streaming=True`, `trust_remote_code=True`). For each article: if its
   `wikipedia_id` is one of the queries' gold ids, keep it as a gold
   `Document`; otherwise keep it as a distractor until `n_distractors` are
   collected. Stop iterating once every gold id has been found *and*
   `n_distractors` distractors are collected. Marked with a `ponytail:`
   comment in code: linear scan with early exit is fine for a handful of
   queries, but if ids are scattered this can still scan a large prefix of
   the dump — a real subset needs an indexed lookup or a pre-filtered local
   KILT dump.
3. `Document.text` = `"\n\n".join(text.paragraph)`, `title=wikipedia_title`,
   `doc_id=wikipedia_id`.
4. Return `(documents, queries)`.

## Testing

- Contract test per new stage: constructing with its config and calling it
  satisfies the relevant `Protocol` (structural check) and its registry
  entry's `config_model`.
- Unit test per stage's own logic (chunk boundaries/overlap, cosine ranking,
  prompt construction) — `embed_texts`/`generate_chat` are mocked via
  `pytest-mock` so stage tests never hit a network call or load a real model.
- `core/clients.py` gets its own tests with the underlying
  `SentenceTransformer`/`OpenAI` clients mocked, verifying latency/token
  plumbing.
- One integration test (`tests/core/test_pipeline.py` or similar) runs
  `run_pipeline` end-to-end over synthetic `Document`/`Query` fixtures with
  both clients mocked — no real API/model calls in the test suite.
- `benchmarks/test_natural_questions.py` mocks `datasets.load_dataset` for
  both configs (small fixed fixture rows) rather than hitting the network in
  CI.

## File layout additions

```
src/ragdoll/core/protocols.py
src/ragdoll/core/clients.py
src/ragdoll/stages/chunkers/fixed_size.py
src/ragdoll/stages/query_transforms/identity.py
src/ragdoll/stages/retrievers/dense.py
src/ragdoll/stages/rerankers/identity.py
src/ragdoll/stages/generators/single_shot.py
tests/core/test_protocols.py          # only if there's non-trivial logic to check; likely folded into test_pipeline.py
tests/core/test_clients.py
tests/core/test_pipeline.py
tests/core/test_runner.py
tests/stages/chunkers/test_fixed_size.py
tests/stages/query_transforms/test_identity.py
tests/stages/retrievers/test_dense.py
tests/stages/rerankers/test_identity.py
tests/stages/generators/test_single_shot.py
tests/benchmarks/test_natural_questions.py
```
