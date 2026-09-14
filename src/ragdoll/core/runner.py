from typing import Any

from ragdoll.core.pipeline import StageCombination
from ragdoll.core.schema import Document, Query, RAGResponse


def expand_grid(
    chunkers: list[str],
    query_transforms: list[str],
    retrievers: list[str],
    rerankers: list[str],
    generators: list[str],
) -> list[StageCombination]:
    raise NotImplementedError


def build_index(chunker: str, retriever: str, documents: list[Document]) -> Any:
    raise NotImplementedError


def run_combination(
    combination: StageCombination, index_handle: Any, queries: list[Query]
) -> list[RAGResponse]:
    raise NotImplementedError
