# Basic RAG Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the scaffolded stubs into one real, working RAG combination:
fixed-size chunking → identity query transform → dense (local-embedding)
retrieval → identity reranking → single-shot LLM generation, runnable
end-to-end against a small real slice of Natural Questions/KILT.

**Architecture:** Each stage is a class constructed with its own Pydantic
config, callable with only the arrow-typed data from AGENTS.md. Two shared
functions in `core/clients.py` are the only code that talks to
`sentence-transformers`/`openai`. `pipeline.py` and `runner.py` dispatch to
whichever stage classes are registered by name, generically.

**Tech Stack:** Python 3.11+, Pydantic v2, `sentence-transformers`, `openai`,
Hugging Face `datasets`, `requests`, `numpy`, pytest + `pytest-mock`.

**Spec:** `docs/superpowers/specs/2026-09-15-basic-rag-pipeline-design.md`

## Global Constraints

- Type-hint everything; `uvx ty check` must be clean, no `# type: ignore`
  without a comment explaining why.
- All cross-stage data uses the Pydantic models in `core/schema.py` exactly
  as defined.
- No stage implementation calls an LLM/embedding provider directly inline —
  only `core/clients.py` imports `sentence_transformers`/`openai`.
- Config for a stage implementation lives in that implementation's own
  Pydantic config model, registered via its stage registry.
- Every stage implementation ships with a contract/unit test before it's
  considered done.
- `uv run pytest`, `uvx ty check`, `uvx ruff check .`, and
  `uvx ruff format --check .` must all pass before the branch is considered
  done (last task in this plan runs all four).
- Work happens on branch `feature/basic-rag-pipeline` (already checked out) —
  never commit to `main`.

---

### Task 1: Stage protocols

**Files:**
- Create: `src/ragdoll/core/protocols.py`
- Test: `tests/core/test_protocols.py`

**Interfaces:**
- Produces: `ChunkerProtocol`, `QueryTransformProtocol`, `RetrieverProtocol`
  (`build_index`, `retrieve`), `RerankerProtocol`, `GeneratorProtocol` — used
  for type hints in `pipeline.py`/`runner.py` from Task 8/9 onward. These are
  `@runtime_checkable` but, since Python's `runtime_checkable` only checks
  that named methods exist (not their signatures), `isinstance` against the
  four single-`__call__` protocols cannot distinguish between them — only
  `RetrieverProtocol` (two required methods) is a meaningful runtime check.
  The test below documents this honestly rather than overclaiming.

- [ ] **Step 1: Write the failing test**

`tests/core/test_protocols.py`:
```python
"""Tests for core.protocols."""

from __future__ import annotations

from ragdoll.core.protocols import (
    ChunkerProtocol,
    GeneratorProtocol,
    QueryTransformProtocol,
    RerankerProtocol,
    RetrieverProtocol,
)


class _Callable:
    def __call__(self, *args: object) -> None:
        return None


class _Retriever:
    def build_index(self, chunks: object) -> None:
        return None

    def retrieve(self, transformed_query: object, index_handle: object) -> None:
        return None


def test_callable_satisfies_single_method_protocols() -> None:
    instance = _Callable()
    assert isinstance(instance, ChunkerProtocol)
    assert isinstance(instance, QueryTransformProtocol)
    assert isinstance(instance, RerankerProtocol)
    assert isinstance(instance, GeneratorProtocol)


def test_plain_object_satisfies_no_protocol() -> None:
    instance = object()
    assert not isinstance(instance, ChunkerProtocol)
    assert not isinstance(instance, RetrieverProtocol)


def test_retriever_protocol_needs_both_methods() -> None:
    assert isinstance(_Retriever(), RetrieverProtocol)
    assert not isinstance(_Callable(), RetrieverProtocol)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/test_protocols.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ragdoll.core.protocols'`

- [ ] **Step 3: Write the implementation**

`src/ragdoll/core/protocols.py`:
```python
"""Structural call shapes every stage implementation must satisfy; see AGENTS.md's Pipeline stages section."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ragdoll.core.schema import (
        Chunk,
        Document,
        IndexHandle,
        Query,
        RAGResponse,
        RetrievedContext,
        TransformedQuery,
    )


@runtime_checkable
class ChunkerProtocol(Protocol):
    """Chunker: list[Document] -> list[Chunk]."""

    def __call__(self, documents: list[Document]) -> list[Chunk]: ...


@runtime_checkable
class QueryTransformProtocol(Protocol):
    """QueryTransform: Query -> TransformedQuery."""

    def __call__(self, query: Query) -> TransformedQuery: ...


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Retrieval's two arrows: BuildIndex and Retrieve, on one implementation."""

    def build_index(self, chunks: list[Chunk]) -> IndexHandle: ...
    def retrieve(self, transformed_query: TransformedQuery, index_handle: IndexHandle) -> list[RetrievedContext]: ...


@runtime_checkable
class RerankerProtocol(Protocol):
    """Rerank: (Query, list[RetrievedContext]) -> list[RetrievedContext]."""

    def __call__(self, query: Query, contexts: list[RetrievedContext]) -> list[RetrievedContext]: ...


@runtime_checkable
class GeneratorProtocol(Protocol):
    """Generate: (Query, list[RetrievedContext]) -> RAGResponse."""

    def __call__(self, query: Query, contexts: list[RetrievedContext]) -> RAGResponse: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/core/test_protocols.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ragdoll/core/protocols.py tests/core/test_protocols.py
git commit -m "Add stage Protocol definitions for generic dispatch"
```

---

### Task 2: Shared client wrapper

**Files:**
- Create: `src/ragdoll/core/clients.py`
- Create: `tests/conftest.py`
- Test: `tests/core/test_clients.py`

**Interfaces:**
- Consumes: `sentence_transformers.SentenceTransformer`, `openai.OpenAI`
  (new deps).
- Produces: `embed_texts(texts: list[str], model_name: str) -> tuple[Any, float]`
  and `generate_chat(messages: list[dict[str, str]], model_name: str, max_tokens: int, temperature: float) -> tuple[str, float, dict[str, int]]`
  — used by `stages/retrievers/dense.py` (Task 6) and
  `stages/generators/single_shot.py` (Task 7).

- [ ] **Step 1: Add dependencies**

```bash
uv add sentence-transformers openai
```

- [ ] **Step 2: Write the failing test**

`tests/conftest.py`:
```python
"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from ragdoll.core.clients import _get_openai_client, _load_embedding_model


@pytest.fixture(autouse=True)
def _clear_client_caches():
    yield
    _load_embedding_model.cache_clear()
    _get_openai_client.cache_clear()
```

`tests/core/test_clients.py`:
```python
"""Tests for core.clients."""

from __future__ import annotations

from ragdoll.core import clients


def test_embed_texts_returns_embeddings_and_latency(mocker):
    fake_model = mocker.Mock()
    fake_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]
    mocker.patch.object(clients, 'SentenceTransformer', return_value=fake_model)

    embeddings, latency_ms = clients.embed_texts(['a', 'b'], 'fake-model')

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert latency_ms >= 0
    fake_model.encode.assert_called_once_with(['a', 'b'])


def test_embed_texts_caches_model_per_name(mocker):
    mock_cls = mocker.patch.object(clients, 'SentenceTransformer')
    mock_cls.return_value.encode.return_value = []

    clients.embed_texts(['a'], 'same-model')
    clients.embed_texts(['b'], 'same-model')

    mock_cls.assert_called_once_with('same-model')


def test_generate_chat_returns_answer_latency_and_token_usage(mocker):
    fake_response = mocker.Mock()
    fake_response.choices = [mocker.Mock(message=mocker.Mock(content='the answer'))]
    fake_response.usage = mocker.Mock(prompt_tokens=10, completion_tokens=5)
    fake_client = mocker.Mock()
    fake_client.chat.completions.create.return_value = fake_response
    mocker.patch.object(clients, 'OpenAI', return_value=fake_client)

    answer, latency_ms, token_usage = clients.generate_chat(
        [{'role': 'user', 'content': 'hi'}],
        'fake-model',
        max_tokens=100,
        temperature=0.0,
    )

    assert answer == 'the answer'
    assert latency_ms >= 0
    assert token_usage == {'prompt_tokens': 10, 'completion_tokens': 5}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/core/test_clients.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ragdoll.core.clients'`

- [ ] **Step 4: Write the implementation**

`src/ragdoll/core/clients.py`:
```python
"""Shared client wrapper for embedding and LLM calls; the only module that imports third-party AI providers."""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

from openai import OpenAI
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=None)
def _load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model_name: str) -> tuple[Any, float]:
    """Embed texts with a local sentence-transformers model; returns (embeddings, latency_ms)."""
    model = _load_embedding_model(model_name)
    start = time.perf_counter()
    embeddings = model.encode(texts)
    latency_ms = (time.perf_counter() - start) * 1000
    return embeddings, latency_ms


@lru_cache(maxsize=None)
def _get_openai_client() -> OpenAI:
    return OpenAI()


def generate_chat(
    messages: list[dict[str, str]],
    model_name: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, float, dict[str, int]]:
    """Call an OpenAI-compatible chat endpoint; returns (answer_text, latency_ms, token_usage)."""
    client = _get_openai_client()
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    token_usage = {
        'prompt_tokens': response.usage.prompt_tokens,
        'completion_tokens': response.usage.completion_tokens,
    }
    return response.choices[0].message.content, latency_ms, token_usage
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/core/test_clients.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/ragdoll/core/clients.py tests/conftest.py tests/core/test_clients.py
git commit -m "Add shared embedding/LLM client wrapper"
```

---

### Task 3: Fixed-size chunker

**Files:**
- Create: `src/ragdoll/stages/chunkers/fixed_size.py`
- Modify: `src/ragdoll/stages/chunkers/__init__.py`
- Test: `tests/stages/chunkers/test_fixed_size.py`

**Interfaces:**
- Consumes: `Document`, `Chunk` from `core/schema.py`; `chunkers` registry
  from `core/registry.py`.
- Produces: `FixedSizeChunker`, `FixedSizeChunkerConfig` (fields
  `chunk_size: int = 500`, `overlap: int = 50`), registered as `"fixed_size"`
  — used by `core/runner.py::build_index` (Task 9) and pipeline tests
  (Task 8).

- [ ] **Step 1: Write the failing test**

`tests/stages/chunkers/test_fixed_size.py`:
```python
"""Tests for FixedSizeChunker."""

from __future__ import annotations

import pytest

from ragdoll.core.schema import Chunk, Document
from ragdoll.stages.chunkers.fixed_size import FixedSizeChunker, FixedSizeChunkerConfig


def test_splits_long_document_into_overlapping_chunks():
    chunker = FixedSizeChunker(FixedSizeChunkerConfig(chunk_size=10, overlap=2))
    documents = [Document(doc_id='d1', text='0123456789abcdefghij')]

    chunks = chunker(documents)

    assert all(isinstance(c, Chunk) for c in chunks)
    assert [c.text for c in chunks] == ['0123456789', '89abcdefgh', 'ghij']
    assert [c.chunk_id for c in chunks] == ['d1::0', 'd1::1', 'd1::2']
    assert [c.position for c in chunks] == [0, 1, 2]
    assert all(c.doc_id == 'd1' for c in chunks)


def test_short_document_produces_single_chunk():
    chunker = FixedSizeChunker(FixedSizeChunkerConfig(chunk_size=500, overlap=50))
    documents = [Document(doc_id='d1', text='short text')]

    chunks = chunker(documents)

    assert len(chunks) == 1
    assert chunks[0].text == 'short text'


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError, match='overlap must be smaller than chunk_size'):
        FixedSizeChunkerConfig(chunk_size=10, overlap=10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/stages/chunkers/test_fixed_size.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ragdoll.stages.chunkers.fixed_size'`

- [ ] **Step 3: Write the implementation**

`src/ragdoll/stages/chunkers/fixed_size.py`:
```python
"""Fixed-size chunker: splits each document into overlapping character windows."""

from __future__ import annotations

from pydantic import BaseModel, model_validator

from ragdoll.core.registry import chunkers
from ragdoll.core.schema import Chunk, Document


class FixedSizeChunkerConfig(BaseModel):
    """Config for FixedSizeChunker."""

    chunk_size: int = 500
    overlap: int = 50

    @model_validator(mode='after')
    def _check_overlap(self) -> FixedSizeChunkerConfig:
        if self.overlap >= self.chunk_size:
            msg = 'overlap must be smaller than chunk_size'
            raise ValueError(msg)
        return self


@chunkers.register('fixed_size', FixedSizeChunkerConfig)
class FixedSizeChunker:
    """Splits each document's text into overlapping fixed-size character windows."""

    def __init__(self, config: FixedSizeChunkerConfig) -> None:
        self._config = config

    def __call__(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        step = self._config.chunk_size - self._config.overlap
        for document in documents:
            position = 0
            start = 0
            while start < len(document.text):
                window = document.text[start : start + self._config.chunk_size]
                chunks.append(
                    Chunk(
                        chunk_id=f'{document.doc_id}::{position}',
                        doc_id=document.doc_id,
                        text=window,
                        position=position,
                    ),
                )
                position += 1
                start += step
        return chunks
```

`src/ragdoll/stages/chunkers/__init__.py`:
```python
"""Chunker implementations: list[Document] -> list[Chunk]."""

from ragdoll.stages.chunkers.fixed_size import FixedSizeChunker, FixedSizeChunkerConfig

__all__ = ['FixedSizeChunker', 'FixedSizeChunkerConfig']
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/stages/chunkers/test_fixed_size.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ragdoll/stages/chunkers tests/stages/chunkers/test_fixed_size.py
git commit -m "Add fixed-size chunker"
```

---

### Task 4: Identity query transform

**Files:**
- Create: `src/ragdoll/stages/query_transforms/identity.py`
- Modify: `src/ragdoll/stages/query_transforms/__init__.py`
- Test: `tests/stages/query_transforms/test_identity.py`

**Interfaces:**
- Consumes: `Query`, `TransformedQuery` from `core/schema.py`;
  `query_transforms` registry.
- Produces: `IdentityQueryTransform`, `IdentityQueryTransformConfig`
  (no fields), registered as `"identity"` — this is `StageCombination`'s
  default `query_transform`, used by every pipeline test from Task 8 on.

- [ ] **Step 1: Write the failing test**

`tests/stages/query_transforms/test_identity.py`:
```python
"""Tests for IdentityQueryTransform."""

from __future__ import annotations

from ragdoll.core.schema import Query, TransformedQuery
from ragdoll.stages.query_transforms.identity import IdentityQueryTransform, IdentityQueryTransformConfig


def test_returns_query_text_as_single_search_text():
    transform = IdentityQueryTransform(IdentityQueryTransformConfig())
    query = Query(query_id='q1', text='what is rag?', lang='en')

    result = transform(query)

    assert isinstance(result, TransformedQuery)
    assert result.query_id == 'q1'
    assert result.search_texts == ['what is rag?']
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/stages/query_transforms/test_identity.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/ragdoll/stages/query_transforms/identity.py`:
```python
"""Identity query transform: passes the query text through unchanged."""

from __future__ import annotations

from pydantic import BaseModel

from ragdoll.core.registry import query_transforms
from ragdoll.core.schema import Query, TransformedQuery


class IdentityQueryTransformConfig(BaseModel):
    """Config for IdentityQueryTransform (no parameters)."""


@query_transforms.register('identity', IdentityQueryTransformConfig)
class IdentityQueryTransform:
    """Passes the query text through as the single search text."""

    def __init__(self, config: IdentityQueryTransformConfig) -> None:
        self._config = config

    def __call__(self, query: Query) -> TransformedQuery:
        return TransformedQuery(query_id=query.query_id, search_texts=[query.text])
```

`src/ragdoll/stages/query_transforms/__init__.py`:
```python
"""Query-transform implementations: Query -> TransformedQuery."""

from ragdoll.stages.query_transforms.identity import IdentityQueryTransform, IdentityQueryTransformConfig

__all__ = ['IdentityQueryTransform', 'IdentityQueryTransformConfig']
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/stages/query_transforms/test_identity.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragdoll/stages/query_transforms tests/stages/query_transforms/test_identity.py
git commit -m "Add identity query transform"
```

---

### Task 5: Identity reranker

**Files:**
- Create: `src/ragdoll/stages/rerankers/identity.py`
- Modify: `src/ragdoll/stages/rerankers/__init__.py`
- Test: `tests/stages/rerankers/test_identity.py`

**Interfaces:**
- Consumes: `Query`, `RetrievedContext` from `core/schema.py`; `rerankers`
  registry.
- Produces: `IdentityReranker`, `IdentityRerankerConfig` (no fields),
  registered as `"identity"` — the default `reranker` in `StageCombination`.

- [ ] **Step 1: Write the failing test**

`tests/stages/rerankers/test_identity.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/stages/rerankers/test_identity.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/ragdoll/stages/rerankers/identity.py`:
```python
"""Identity reranker: returns the retrieved contexts unchanged."""

from __future__ import annotations

from pydantic import BaseModel

from ragdoll.core.registry import rerankers
from ragdoll.core.schema import Query, RetrievedContext


class IdentityRerankerConfig(BaseModel):
    """Config for IdentityReranker (no parameters)."""


@rerankers.register('identity', IdentityRerankerConfig)
class IdentityReranker:
    """Passes retrieved contexts through unchanged."""

    def __init__(self, config: IdentityRerankerConfig) -> None:
        self._config = config

    def __call__(self, _query: Query, contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        return contexts
```

`src/ragdoll/stages/rerankers/__init__.py`:
```python
"""Reranker implementations: (Query, list[RetrievedContext]) -> list[RetrievedContext]."""

from ragdoll.stages.rerankers.identity import IdentityReranker, IdentityRerankerConfig

__all__ = ['IdentityReranker', 'IdentityRerankerConfig']
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/stages/rerankers/test_identity.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragdoll/stages/rerankers tests/stages/rerankers/test_identity.py
git commit -m "Add identity reranker"
```

---

### Task 6: Dense retriever

**Files:**
- Create: `src/ragdoll/stages/retrievers/dense.py`
- Modify: `src/ragdoll/stages/retrievers/__init__.py`
- Test: `tests/stages/retrievers/test_dense.py`

**Interfaces:**
- Consumes: `embed_texts` from `core/clients.py` (Task 2); `IndexHandle`,
  `Chunk`, `TransformedQuery`, `RetrievedContext` from `core/schema.py`;
  `retrievers` registry.
- Produces: `DenseRetriever`, `DenseRetrieverConfig` (fields
  `embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'`,
  `top_k: int = 5`), `DenseIndex` (dataclass: `chunks: list[Chunk]`,
  `embeddings: np.ndarray`), registered as `"dense"` — used by
  `core/runner.py::build_index` (Task 9) and pipeline/runner tests.

- [ ] **Step 1: Add dependency**

```bash
uv add numpy
```

- [ ] **Step 2: Write the failing test**

`tests/stages/retrievers/test_dense.py`:
```python
"""Tests for DenseRetriever."""

from __future__ import annotations

import numpy as np

from ragdoll.core.schema import Chunk, TransformedQuery
from ragdoll.stages.retrievers.dense import DenseRetriever, DenseRetrieverConfig


def test_build_index_embeds_chunk_texts(mocker):
    mock_embed = mocker.patch(
        'ragdoll.stages.retrievers.dense.embed_texts',
        return_value=(np.array([[1.0, 0.0], [0.0, 1.0]]), 5.0),
    )
    chunks = [
        Chunk(chunk_id='c1', doc_id='d1', text='alpha', position=0),
        Chunk(chunk_id='c2', doc_id='d1', text='beta', position=1),
    ]
    retriever = DenseRetriever(DenseRetrieverConfig())

    index_handle = retriever.build_index(chunks)

    mock_embed.assert_called_once_with(['alpha', 'beta'], DenseRetrieverConfig().embedding_model)
    assert index_handle.chunks == chunks


def test_retrieve_ranks_by_cosine_similarity(mocker):
    mocker.patch(
        'ragdoll.stages.retrievers.dense.embed_texts',
        side_effect=[
            (np.array([[1.0, 0.0], [0.0, 1.0]]), 5.0),
            (np.array([[0.9, 0.1]]), 2.0),
        ],
    )
    chunks = [
        Chunk(chunk_id='c1', doc_id='d1', text='alpha', position=0),
        Chunk(chunk_id='c2', doc_id='d1', text='beta', position=1),
    ]
    retriever = DenseRetriever(DenseRetrieverConfig(top_k=2))
    index_handle = retriever.build_index(chunks)

    results = retriever.retrieve(TransformedQuery(query_id='q1', search_texts=['query text']), index_handle)

    assert [r.chunk_id for r in results] == ['c1', 'c2']
    assert [r.rank for r in results] == [1, 2]
    assert results[0].score > results[1].score
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/stages/retrievers/test_dense.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Write the implementation**

`src/ragdoll/stages/retrievers/dense.py`:
```python
"""Dense retriever: local sentence-transformers embeddings + brute-force cosine similarity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

from ragdoll.core.clients import embed_texts
from ragdoll.core.registry import retrievers
from ragdoll.core.schema import IndexHandle, RetrievedContext

if TYPE_CHECKING:
    from ragdoll.core.schema import Chunk, TransformedQuery


class DenseRetrieverConfig(BaseModel):
    """Config for DenseRetriever."""

    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    top_k: int = 5


@dataclass
class DenseIndex:
    """Opaque IndexHandle for DenseRetriever: chunks plus their embeddings, in matching order."""

    chunks: list[Chunk]
    embeddings: np.ndarray


@retrievers.register('dense', DenseRetrieverConfig)
class DenseRetriever:
    """Embeds chunks and queries locally, retrieves by cosine similarity."""

    def __init__(self, config: DenseRetrieverConfig) -> None:
        self._config = config

    def build_index(self, chunks: list[Chunk]) -> IndexHandle:
        embeddings, _latency_ms = embed_texts([chunk.text for chunk in chunks], self._config.embedding_model)
        return DenseIndex(chunks=chunks, embeddings=np.asarray(embeddings))

    def retrieve(self, transformed_query: TransformedQuery, index_handle: IndexHandle) -> list[RetrievedContext]:
        assert isinstance(index_handle, DenseIndex)
        query_text = ' '.join(transformed_query.search_texts)
        query_embedding, _latency_ms = embed_texts([query_text], self._config.embedding_model)
        query_vector = np.asarray(query_embedding)[0]

        chunk_norms = np.linalg.norm(index_handle.embeddings, axis=1)
        query_norm = np.linalg.norm(query_vector)
        scores = index_handle.embeddings @ query_vector / (chunk_norms * query_norm + 1e-10)

        ranked_indices = np.argsort(-scores)[: self._config.top_k]
        return [
            RetrievedContext(
                chunk_id=index_handle.chunks[i].chunk_id,
                doc_id=index_handle.chunks[i].doc_id,
                text=index_handle.chunks[i].text,
                score=float(scores[i]),
                rank=rank,
            )
            for rank, i in enumerate(ranked_indices, start=1)
        ]
```

`src/ragdoll/stages/retrievers/__init__.py`:
```python
"""Retriever implementations: index building and search."""

from ragdoll.stages.retrievers.dense import DenseIndex, DenseRetriever, DenseRetrieverConfig

__all__ = ['DenseIndex', 'DenseRetriever', 'DenseRetrieverConfig']
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/stages/retrievers/test_dense.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/ragdoll/stages/retrievers tests/stages/retrievers/test_dense.py
git commit -m "Add dense retriever"
```

---

### Task 7: Single-shot generator

**Files:**
- Create: `src/ragdoll/stages/generators/single_shot.py`
- Modify: `src/ragdoll/stages/generators/__init__.py`
- Test: `tests/stages/generators/test_single_shot.py`

**Interfaces:**
- Consumes: `generate_chat` from `core/clients.py` (Task 2); `Query`,
  `RetrievedContext`, `RAGResponse` from `core/schema.py`; `generators`
  registry.
- Produces: `SingleShotGenerator`, `SingleShotGeneratorConfig` (fields
  `model_name: str = 'gpt-4o-mini'`, `max_tokens: int = 512`,
  `temperature: float = 0.0`), registered as `"single_shot"` — used as the
  `generator` in every `StageCombination` from Task 8 on.

- [ ] **Step 1: Write the failing test**

`tests/stages/generators/test_single_shot.py`:
```python
"""Tests for SingleShotGenerator."""

from __future__ import annotations

from ragdoll.core.schema import Query, RAGResponse, RetrievedContext
from ragdoll.stages.generators.single_shot import SingleShotGenerator, SingleShotGeneratorConfig


def test_builds_prompt_and_wraps_response(mocker):
    mock_generate = mocker.patch(
        'ragdoll.stages.generators.single_shot.generate_chat',
        return_value=('the answer', 42.0, {'prompt_tokens': 10, 'completion_tokens': 3}),
    )
    query = Query(query_id='q1', text='what is rag?', lang='en')
    contexts = [
        RetrievedContext(chunk_id='c1', doc_id='d1', text='RAG stands for retrieval-augmented generation.', rank=1),
    ]
    generator = SingleShotGenerator(SingleShotGeneratorConfig())

    response = generator(query, contexts)

    assert isinstance(response, RAGResponse)
    assert response.query_id == 'q1'
    assert response.answer == 'the answer'
    assert response.retrieved_contexts == contexts
    assert response.latency_ms == 42.0
    assert response.token_usage == {'prompt_tokens': 10, 'completion_tokens': 3}

    messages = mock_generate.call_args.args[0]
    assert 'what is rag?' in messages[0]['content']
    assert 'RAG stands for retrieval-augmented generation.' in messages[0]['content']
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/stages/generators/test_single_shot.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/ragdoll/stages/generators/single_shot.py`:
```python
"""Single-shot generator: one LLM call over the query and its retrieved contexts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from ragdoll.core.clients import generate_chat
from ragdoll.core.registry import generators
from ragdoll.core.schema import RAGResponse

if TYPE_CHECKING:
    from ragdoll.core.schema import Query, RetrievedContext


class SingleShotGeneratorConfig(BaseModel):
    """Config for SingleShotGenerator."""

    model_name: str = 'gpt-4o-mini'
    max_tokens: int = 512
    temperature: float = 0.0


@generators.register('single_shot', SingleShotGeneratorConfig)
class SingleShotGenerator:
    """Answers a query in a single LLM call, grounded in its retrieved contexts."""

    def __init__(self, config: SingleShotGeneratorConfig) -> None:
        self._config = config

    def __call__(self, query: Query, contexts: list[RetrievedContext]) -> RAGResponse:
        context_block = '\n\n'.join(f'[{i}] {context.text}' for i, context in enumerate(contexts, start=1))
        prompt = (
            f'Answer the question using only the numbered contexts below.\n\n'
            f'Contexts:\n{context_block}\n\nQuestion: {query.text}\nAnswer:'
        )
        answer, latency_ms, token_usage = generate_chat(
            [{'role': 'user', 'content': prompt}],
            self._config.model_name,
            self._config.max_tokens,
            self._config.temperature,
        )
        return RAGResponse(
            query_id=query.query_id,
            answer=answer,
            retrieved_contexts=contexts,
            latency_ms=latency_ms,
            token_usage=token_usage,
        )
```

`src/ragdoll/stages/generators/__init__.py`:
```python
"""Generator implementations: (Query, list[RetrievedContext]) -> RAGResponse."""

from ragdoll.stages.generators.single_shot import SingleShotGenerator, SingleShotGeneratorConfig

__all__ = ['SingleShotGenerator', 'SingleShotGeneratorConfig']
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/stages/generators/test_single_shot.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragdoll/stages/generators tests/stages/generators/test_single_shot.py
git commit -m "Add single-shot generator"
```

---

### Task 8: Pipeline wiring

**Files:**
- Modify: `src/ragdoll/core/pipeline.py`
- Modify: `src/ragdoll/stages/__init__.py`
- Test: `tests/core/test_pipeline.py`

**Interfaces:**
- Consumes: registries from `core/registry.py`; all five stage
  implementations (Tasks 3-7); `StageCombination` (already defined in
  `pipeline.py`).
- Produces: `run_pipeline(combination: StageCombination, index_handle: IndexHandle, query: Query) -> RAGResponse`
  — used by `core/runner.py::run_combination` (Task 9).

- [ ] **Step 1: Write the failing test**

`tests/core/test_pipeline.py`:
```python
"""Integration test: full pipeline through registered real stages, only network/model calls mocked."""

from __future__ import annotations

import ragdoll.stages  # noqa: F401  (registers every stage implementation)
from ragdoll.core.pipeline import StageCombination, run_pipeline
from ragdoll.core.registry import retrievers
from ragdoll.core.schema import Chunk, Query


def test_run_pipeline_end_to_end(mocker):
    mocker.patch('ragdoll.stages.retrievers.dense.embed_texts', return_value=([[0.1, 0.2]], 1.0))
    mocker.patch(
        'ragdoll.stages.generators.single_shot.generate_chat',
        return_value=('an answer', 3.0, {'prompt_tokens': 1, 'completion_tokens': 1}),
    )

    retriever_entry = retrievers.get('dense')
    retriever = retriever_entry.cls(retriever_entry.config_model())
    chunks = [Chunk(chunk_id='c1', doc_id='d1', text='some context', position=0)]
    index_handle = retriever.build_index(chunks)

    combination = StageCombination(chunker='fixed_size', retriever='dense', generator='single_shot')
    query = Query(query_id='q1', text='what is rag?', lang='en')

    response = run_pipeline(combination, index_handle, query)

    assert response.query_id == 'q1'
    assert response.answer == 'an answer'
    assert len(response.retrieved_contexts) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/test_pipeline.py -v`
Expected: FAIL — `run_pipeline` still raises `NotImplementedError`

- [ ] **Step 3: Write the implementation**

`src/ragdoll/stages/__init__.py`:
```python
"""Stage implementations: chunkers, query transforms, retrievers, rerankers, and generators."""

from ragdoll.stages import chunkers, generators, query_transforms, rerankers, retrievers

__all__ = ['chunkers', 'generators', 'query_transforms', 'rerankers', 'retrievers']
```

In `src/ragdoll/core/pipeline.py`, replace the body of `run_pipeline`
(keep `StageCombination` and the module docstring/imports as they are,
just add the registry import and implement the function):

```python
"""Composes stage callables into one run for a given stage combination."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ragdoll.core.registry import generators, query_transforms, rerankers, retrievers

if TYPE_CHECKING:
    from ragdoll.core.schema import IndexHandle, Query, RAGResponse


class StageCombination(BaseModel):
    """Identifies one full combination of stage implementations and their configs."""

    chunker: str
    chunker_config: dict[str, Any] = {}
    query_transform: str = 'identity'
    query_transform_config: dict[str, Any] = {}
    retriever: str
    retriever_config: dict[str, Any] = {}
    reranker: str = 'identity'
    reranker_config: dict[str, Any] = {}
    generator: str
    generator_config: dict[str, Any] = {}


def run_pipeline(combination: StageCombination, index_handle: IndexHandle, query: Query) -> RAGResponse:
    """Run one query through a stage combination against a built index."""
    qt_entry = query_transforms.get(combination.query_transform)
    query_transform = qt_entry.cls(qt_entry.config_model(**combination.query_transform_config))
    transformed_query = query_transform(query)

    retriever_entry = retrievers.get(combination.retriever)
    retriever = retriever_entry.cls(retriever_entry.config_model(**combination.retriever_config))
    retrieved_contexts = retriever.retrieve(transformed_query, index_handle)

    reranker_entry = rerankers.get(combination.reranker)
    reranker = reranker_entry.cls(reranker_entry.config_model(**combination.reranker_config))
    reranked_contexts = reranker(query, retrieved_contexts)

    generator_entry = generators.get(combination.generator)
    generator = generator_entry.cls(generator_entry.config_model(**combination.generator_config))
    return generator(query, reranked_contexts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/core/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragdoll/core/pipeline.py src/ragdoll/stages/__init__.py tests/core/test_pipeline.py
git commit -m "Wire run_pipeline to dispatch through the stage registries"
```

---

### Task 9: Runner wiring

**Files:**
- Modify: `src/ragdoll/core/runner.py`
- Test: `tests/core/test_runner.py`

**Interfaces:**
- Consumes: `chunkers`, `retrievers` registries; `run_pipeline` (Task 8);
  `StageCombination`.
- Produces: `expand_grid(...) -> list[StageCombination]`,
  `build_index(chunker: str, retriever: str, documents: list[Document], chunker_config: dict[str, Any] | None = None, retriever_config: dict[str, Any] | None = None) -> IndexHandle`
  (signature change from the stub — adds the two config dicts),
  `run_combination(combination: StageCombination, index_handle: IndexHandle, queries: list[Query]) -> list[RAGResponse]`.

- [ ] **Step 1: Write the failing test**

`tests/core/test_runner.py`:
```python
"""Tests for core.runner."""

from __future__ import annotations

import ragdoll.stages  # noqa: F401  (registers every stage implementation)
from ragdoll.core.pipeline import StageCombination
from ragdoll.core.runner import build_index, expand_grid, run_combination
from ragdoll.core.schema import Document, Query


def test_expand_grid_produces_cartesian_product():
    combinations = expand_grid(
        chunkers=['fixed_size'],
        query_transforms=['identity'],
        retrievers=['dense'],
        rerankers=['identity'],
        generators=['single_shot'],
    )

    assert combinations == [
        StageCombination(
            chunker='fixed_size',
            query_transform='identity',
            retriever='dense',
            reranker='identity',
            generator='single_shot',
        ),
    ]


def test_build_index_chunks_and_embeds_documents(mocker):
    mock_embed = mocker.patch('ragdoll.stages.retrievers.dense.embed_texts', return_value=([[0.1, 0.2]], 1.0))
    documents = [Document(doc_id='d1', text='short doc')]

    index_handle = build_index('fixed_size', 'dense', documents, chunker_config={'chunk_size': 500, 'overlap': 50})

    assert len(index_handle.chunks) == 1
    mock_embed.assert_called_once()


def test_run_combination_runs_every_query(mocker):
    mocker.patch('ragdoll.stages.retrievers.dense.embed_texts', return_value=([[0.1, 0.2]], 1.0))
    mocker.patch(
        'ragdoll.stages.generators.single_shot.generate_chat',
        return_value=('an answer', 3.0, {'prompt_tokens': 1, 'completion_tokens': 1}),
    )
    documents = [Document(doc_id='d1', text='short doc')]
    index_handle = build_index('fixed_size', 'dense', documents)
    combination = StageCombination(chunker='fixed_size', retriever='dense', generator='single_shot')
    queries = [Query(query_id='q1', text='q1?', lang='en'), Query(query_id='q2', text='q2?', lang='en')]

    responses = run_combination(combination, index_handle, queries)

    assert [r.query_id for r in responses] == ['q1', 'q2']
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/test_runner.py -v`
Expected: FAIL — all three functions still raise `NotImplementedError`

- [ ] **Step 3: Write the implementation**

`src/ragdoll/core/runner.py`:
```python
"""Expands a combination grid, builds/reuses indexes, and runs pipelines."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Any

from ragdoll.core.pipeline import StageCombination, run_pipeline
from ragdoll.core.registry import chunkers, retrievers

if TYPE_CHECKING:
    from ragdoll.core.schema import Document, IndexHandle, Query, RAGResponse


def expand_grid(
    chunkers: list[str],
    query_transforms: list[str],
    retrievers: list[str],
    rerankers: list[str],
    generators: list[str],
) -> list[StageCombination]:
    """Enumerate every stage combination across the given per-slot options."""
    return [
        StageCombination(chunker=c, query_transform=qt, retriever=r, reranker=rr, generator=g)
        for c, qt, r, rr, g in product(chunkers, query_transforms, retrievers, rerankers, generators)
    ]


def build_index(
    chunker: str,
    retriever: str,
    documents: list[Document],
    chunker_config: dict[str, Any] | None = None,
    retriever_config: dict[str, Any] | None = None,
) -> IndexHandle:
    """Build and return the index handle for one (chunker, retriever) pair."""
    chunker_entry = chunkers.get(chunker)
    chunker_instance = chunker_entry.cls(chunker_entry.config_model(**(chunker_config or {})))
    chunks = chunker_instance(documents)

    retriever_entry = retrievers.get(retriever)
    retriever_instance = retriever_entry.cls(retriever_entry.config_model(**(retriever_config or {})))
    return retriever_instance.build_index(chunks)


def run_combination(
    combination: StageCombination,
    index_handle: IndexHandle,
    queries: list[Query],
) -> list[RAGResponse]:
    """Run every query through one stage combination against a built index."""
    return [run_pipeline(combination, index_handle, query) for query in queries]
```

Note: this shadows the module-level `chunkers`/`retrievers` registry imports
with `expand_grid`'s own parameter names of the same name — harmless
(function-local shadowing only affects `expand_grid`'s own body, which
doesn't use the registries), and keeps `expand_grid`'s parameter names
consistent with `StageCombination`'s field names.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/core/test_runner.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ragdoll/core/runner.py tests/core/test_runner.py
git commit -m "Wire runner's expand_grid/build_index/run_combination"
```

---

### Task 10: Natural Questions (KILT) benchmark adapter

**Files:**
- Create/modify: `src/ragdoll/benchmarks/natural_questions.py`
- Test: `tests/benchmarks/test_natural_questions.py`

**Interfaces:**
- Consumes: `Document`, `Query` from `core/schema.py`.
- Produces: `load_natural_questions(split: str = 'validation', n_queries: int = 5, n_distractors: int = 20) -> tuple[list[Document], list[Query]]`.

- [ ] **Step 1: Add dependencies**

```bash
uv add datasets requests
```

- [ ] **Step 2: Write the failing test**

`tests/benchmarks/test_natural_questions.py`:
```python
"""Tests for the Natural Questions (KILT) benchmark adapter."""

from __future__ import annotations

import json

from ragdoll.benchmarks.natural_questions import load_natural_questions


def _fake_task_rows():
    return [
        {
            'id': 'q1',
            'input': 'what is the therefore sign?',
            'output': [
                {'answer': 'the therefore sign', 'meta': {'score': -1}, 'provenance': [{'wikipedia_id': '10593264'}]},
                {'answer': '', 'meta': {'score': -1}, 'provenance': []},
            ],
        },
    ]


def _fake_wiki_lines():
    lines = [
        {'wikipedia_id': '10593264', 'wikipedia_title': 'Therefore sign', 'text': ['Gold paragraph.\n']},
        {'wikipedia_id': '1', 'wikipedia_title': 'Distractor One', 'text': ['Distractor text.\n']},
        {'wikipedia_id': '2', 'wikipedia_title': 'Distractor Two', 'text': ['More distractor text.\n']},
    ]
    return [json.dumps(line).encode('utf-8') for line in lines]


def test_load_natural_questions_returns_queries_and_documents(mocker):
    mocker.patch(
        'ragdoll.benchmarks.natural_questions.datasets.load_dataset',
        return_value=_fake_task_rows(),
    )
    fake_response = mocker.Mock()
    fake_response.iter_lines.return_value = _fake_wiki_lines()
    mocker.patch('ragdoll.benchmarks.natural_questions.requests.get', return_value=fake_response)

    documents, queries = load_natural_questions(n_queries=1, n_distractors=2)

    assert len(queries) == 1
    assert queries[0].query_id == 'q1'
    assert queries[0].gold_answers == ['the therefore sign']
    assert queries[0].gold_doc_ids == ['10593264']

    doc_ids = {d.doc_id for d in documents}
    assert doc_ids == {'10593264', '1', '2'}
    gold_doc = next(d for d in documents if d.doc_id == '10593264')
    assert gold_doc.text == 'Gold paragraph.\n'
    assert gold_doc.title == 'Therefore sign'
    fake_response.close.assert_called_once()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/benchmarks/test_natural_questions.py -v`
Expected: FAIL — `load_natural_questions` doesn't exist yet

- [ ] **Step 4: Write the implementation**

`src/ragdoll/benchmarks/natural_questions.py`:
```python
"""Natural Questions (KILT format) benchmark adapter.

facebook/kilt_wikipedia can't be loaded through the `datasets` library
(its script-based loader is no longer supported by current `datasets`
versions), so the knowledge source is streamed directly from the URL that
loader used to download.
"""

from __future__ import annotations

import json

import datasets
import requests

from ragdoll.core.schema import Document, Query

_TASKS_REPO = 'facebook/kilt_tasks'
_KNOWLEDGE_SOURCE_URL = 'http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json'


def load_natural_questions(
    split: str = 'validation',
    n_queries: int = 5,
    n_distractors: int = 20,
) -> tuple[list[Document], list[Query]]:
    """Load n_queries real NQ/KILT queries plus their gold documents and n_distractors distractors."""
    task_rows = datasets.load_dataset(_TASKS_REPO, 'nq', split=f'{split}[:{n_queries}]')

    queries: list[Query] = []
    gold_ids: set[str] = set()
    for row in task_rows:
        answers = [output['answer'] for output in row['output'] if output['answer']]
        doc_ids = {provenance['wikipedia_id'] for output in row['output'] for provenance in output['provenance']}
        gold_ids |= doc_ids
        queries.append(
            Query(
                query_id=row['id'],
                text=row['input'],
                lang='en',
                gold_answers=answers,
                gold_doc_ids=sorted(doc_ids),
            ),
        )

    documents = _collect_documents(gold_ids, n_distractors)
    return documents, queries


def _collect_documents(gold_ids: set[str], n_distractors: int) -> list[Document]:
    documents: list[Document] = []
    found_gold_ids: set[str] = set()
    distractor_count = 0

    response = requests.get(_KNOWLEDGE_SOURCE_URL, stream=True, timeout=30)
    try:
        # ponytail: linear scan with early exit -- fine for a handful of
        # queries; if gold ids are scattered this can still scan a large
        # prefix of the dump. A real subset needs an indexed lookup or a
        # pre-filtered local KILT dump.
        for line in response.iter_lines():
            if not line:
                continue
            article = json.loads(line)
            is_gold = article['wikipedia_id'] in gold_ids
            if not is_gold and distractor_count >= n_distractors:
                continue
            documents.append(
                Document(
                    doc_id=article['wikipedia_id'],
                    text=''.join(article['text']),
                    title=article['wikipedia_title'],
                ),
            )
            if is_gold:
                found_gold_ids.add(article['wikipedia_id'])
            else:
                distractor_count += 1
            if found_gold_ids == gold_ids and distractor_count >= n_distractors:
                break
    finally:
        response.close()
    return documents
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/benchmarks/test_natural_questions.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/ragdoll/benchmarks/natural_questions.py tests/benchmarks/test_natural_questions.py
git commit -m "Add Natural Questions (KILT) benchmark adapter"
```

---

### Task 11: Full verification pass

**Files:** none (verification only; fix whatever these commands flag)

- [ ] **Step 1: Format**

Run: `uv run ruff format .`
Expected: reformats files if needed, exits 0

- [ ] **Step 2: Lint**

Run: `uv run ruff check .`
Expected: `All checks passed!` — fix any reported issues and re-run

- [ ] **Step 3: Type check**

Run: `uvx ty check`
Expected: no errors — fix any reported issues and re-run

- [ ] **Step 4: Full test suite**

Run: `uv run pytest`
Expected: all tests pass, including every test from Tasks 1-10

- [ ] **Step 5: Commit any fixes from steps 1-4**

```bash
git add -A
git commit -m "Fix lint/type/format issues from full verification pass"
```

(Skip this commit if steps 1-4 required no changes.)

---

## Self-Review Notes

- **Spec coverage:** every section of the design spec
  (`2026-09-15-basic-rag-pipeline-design.md`) maps to a task: protocols →
  Task 1, clients → Task 2, the five stages → Tasks 3-7, pipeline → Task 8,
  runner → Task 9, benchmark adapter → Task 10, the four required checks →
  Task 11.
- **Type consistency:** `DenseIndex`/`embed_texts`/`generate_chat` signatures
  are identical everywhere they're referenced across Tasks 2, 6, 7, 8, 9.
  `build_index`'s new config-dict parameters (Task 9) match how Task 8's
  `run_pipeline` already reads `combination.<slot>_config` dicts.
- **Known runtime limitation, called out rather than hidden:** the
  single-`__call__` Protocols in Task 1 can't be told apart by `isinstance`
  at runtime (Python's `runtime_checkable` only checks member names) — the
  real per-stage contract checks are the output-type assertions in each
  stage's own unit test (Tasks 3-7), not Protocol `isinstance` checks.
