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
