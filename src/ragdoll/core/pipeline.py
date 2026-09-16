"""Composes stage callables into one run for a given stage combination."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ragdoll.core.registry import generators, query_transforms, rerankers, retrievers

if TYPE_CHECKING:
    from ragdoll.core.protocols import (
        GeneratorProtocol,
        QueryTransformProtocol,
        RerankerProtocol,
        RetrieverProtocol,
    )
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


@dataclass
class PipelineStages:
    """One instantiated stage per non-index slot, reusable across every query in a combination."""

    query_transform: QueryTransformProtocol
    retriever: RetrieverProtocol
    reranker: RerankerProtocol
    generator: GeneratorProtocol


def build_pipeline_stages(combination: StageCombination) -> PipelineStages:
    """Instantiate the query-transform, retriever, reranker, and generator for a combination."""
    qt_entry = query_transforms.get(combination.query_transform)
    retriever_entry = retrievers.get(combination.retriever)
    reranker_entry = rerankers.get(combination.reranker)
    generator_entry = generators.get(combination.generator)
    return PipelineStages(
        query_transform=qt_entry.cls(qt_entry.config_model(**combination.query_transform_config)),
        retriever=retriever_entry.cls(retriever_entry.config_model(**combination.retriever_config)),
        reranker=reranker_entry.cls(reranker_entry.config_model(**combination.reranker_config)),
        generator=generator_entry.cls(generator_entry.config_model(**combination.generator_config)),
    )


def run_pipeline_stages(stages: PipelineStages, index_handle: IndexHandle, query: Query) -> RAGResponse:
    """Run one query through already-instantiated stages against a built index."""
    pre_generation_start = time.perf_counter()

    transformed_query = stages.query_transform(query)
    retrieved_contexts = stages.retriever.retrieve(transformed_query, index_handle)
    reranked_contexts = stages.reranker(query, retrieved_contexts)

    pre_generation_latency_ms = (time.perf_counter() - pre_generation_start) * 1000

    response = stages.generator(query, reranked_contexts)
    response.latency_ms += pre_generation_latency_ms
    return response


def run_pipeline(combination: StageCombination, index_handle: IndexHandle, query: Query) -> RAGResponse:
    """Run one query through a stage combination against a built index."""
    return run_pipeline_stages(build_pipeline_stages(combination), index_handle, query)
