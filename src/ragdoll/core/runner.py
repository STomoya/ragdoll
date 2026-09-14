"""Expands a combination grid, builds/reuses indexes, and runs pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragdoll.core.pipeline import StageCombination
    from ragdoll.core.schema import Document, IndexHandle, Query, RAGResponse


def expand_grid(
    chunkers: list[str],
    query_transforms: list[str],
    retrievers: list[str],
    rerankers: list[str],
    generators: list[str],
) -> list[StageCombination]:
    """Enumerate every stage combination across the given per-slot options."""
    raise NotImplementedError


def build_index(chunker: str, retriever: str, documents: list[Document]) -> IndexHandle:
    """Build and return the index handle for one (chunker, retriever) pair."""
    raise NotImplementedError


def run_combination(
    combination: StageCombination,
    index_handle: IndexHandle,
    queries: list[Query],
) -> list[RAGResponse]:
    """Run every query through one stage combination against a built index."""
    raise NotImplementedError
