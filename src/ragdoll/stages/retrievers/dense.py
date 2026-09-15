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
        """Build a dense index from chunks by embedding their text."""
        embeddings, _latency_ms = embed_texts([chunk.text for chunk in chunks], self._config.embedding_model)
        return DenseIndex(chunks=chunks, embeddings=np.asarray(embeddings))

    def retrieve(self, transformed_query: TransformedQuery, index_handle: IndexHandle) -> list[RetrievedContext]:
        """Retrieve chunks by cosine similarity of their embeddings to the query embedding."""
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
