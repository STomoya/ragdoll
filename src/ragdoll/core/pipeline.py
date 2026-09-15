"""Composes stage callables into one run for a given stage combination."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

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
    raise NotImplementedError
