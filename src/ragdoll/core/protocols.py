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

    def __call__(self, documents: list[Document]) -> list[Chunk]: ...  # noqa: D102


@runtime_checkable
class QueryTransformProtocol(Protocol):
    """QueryTransform: Query -> TransformedQuery."""

    def __call__(self, query: Query) -> TransformedQuery: ...  # noqa: D102


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Retrieval's two arrows: BuildIndex and Retrieve, on one implementation."""

    def build_index(self, chunks: list[Chunk]) -> IndexHandle: ...  # noqa: D102
    def retrieve(self, transformed_query: TransformedQuery, index_handle: IndexHandle) -> list[RetrievedContext]: ...  # noqa: D102


@runtime_checkable
class RerankerProtocol(Protocol):
    """Rerank: (Query, list[RetrievedContext]) -> list[RetrievedContext]."""

    def __call__(self, query: Query, contexts: list[RetrievedContext]) -> list[RetrievedContext]: ...  # noqa: D102


@runtime_checkable
class GeneratorProtocol(Protocol):
    """Generate: (Query, list[RetrievedContext]) -> RAGResponse."""

    def __call__(self, query: Query, contexts: list[RetrievedContext]) -> RAGResponse: ...  # noqa: D102
