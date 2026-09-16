"""Expands a combination grid, builds/reuses indexes, and runs pipelines."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Any

from ragdoll.core.pipeline import StageCombination, build_pipeline_stages, run_pipeline_stages
from ragdoll.core.registry import chunkers, incompatibility_reason, is_compatible, retrievers

if TYPE_CHECKING:
    from ragdoll.core.schema import Document, IndexHandle, Query, RAGResponse


class IncompatibleStagesError(ValueError):
    """Raised when build_index is called with a declared-incompatible chunker/retriever pair."""

    def __init__(self, chunker_name: str, retriever_name: str, reason: str | None) -> None:
        super().__init__(f"'{chunker_name}' is incompatible with '{retriever_name}': {reason}")


def expand_grid(
    chunkers: list[str],
    query_transforms: list[str],
    retrievers: list[str],
    rerankers: list[str],
    generators: list[str],
) -> list[StageCombination]:
    """Enumerate every stage combination across the given per-slot options.

    Combinations whose (chunker, retriever) pair is declared incompatible in
    the registry are excluded rather than raising — see AGENTS.md.
    """
    return [
        StageCombination(chunker=c, query_transform=qt, retriever=r, reranker=rr, generator=g)
        for c, qt, r, rr, g in product(chunkers, query_transforms, retrievers, rerankers, generators)
        if is_compatible(c, r)
    ]


def build_index(
    chunker: str,
    retriever: str,
    documents: list[Document],
    chunker_config: dict[str, Any] | None = None,
    retriever_config: dict[str, Any] | None = None,
) -> IndexHandle:
    """Build and return the index handle for one (chunker, retriever) pair."""
    if not is_compatible(chunker, retriever):
        raise IncompatibleStagesError(chunker, retriever, incompatibility_reason(chunker, retriever))

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
    stages = build_pipeline_stages(combination)
    return [run_pipeline_stages(stages, index_handle, query) for query in queries]
