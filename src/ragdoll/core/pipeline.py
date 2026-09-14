from typing import Any

from pydantic import BaseModel

from ragdoll.core.schema import Query, RAGResponse


class StageCombination(BaseModel):
    chunker: str
    chunker_config: dict[str, Any] = {}
    query_transform: str = "identity"
    query_transform_config: dict[str, Any] = {}
    retriever: str
    retriever_config: dict[str, Any] = {}
    reranker: str = "identity"
    reranker_config: dict[str, Any] = {}
    generator: str
    generator_config: dict[str, Any] = {}


def run_pipeline(
    combination: StageCombination, index_handle: Any, query: Query
) -> RAGResponse:
    raise NotImplementedError
