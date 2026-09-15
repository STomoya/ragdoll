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
