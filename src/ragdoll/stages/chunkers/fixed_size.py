"""Fixed-size chunker: splits each document into overlapping character windows."""

from __future__ import annotations

from pydantic import BaseModel, model_validator

from ragdoll.core.registry import chunkers
from ragdoll.core.schema import Chunk, Document


class FixedSizeChunkerConfig(BaseModel):
    """Config for FixedSizeChunker."""

    chunk_size: int = 500
    overlap: int = 50

    @model_validator(mode='after')
    def _check_overlap(self) -> FixedSizeChunkerConfig:
        if self.overlap >= self.chunk_size:
            msg = 'overlap must be smaller than chunk_size'
            raise ValueError(msg)
        return self


@chunkers.register('fixed_size', FixedSizeChunkerConfig)
class FixedSizeChunker:
    """Splits each document's text into overlapping fixed-size character windows."""

    def __init__(self, config: FixedSizeChunkerConfig) -> None:
        self._config = config

    def __call__(self, documents: list[Document]) -> list[Chunk]:
        """Split documents into overlapping chunks."""
        chunks: list[Chunk] = []
        step = self._config.chunk_size - self._config.overlap
        for document in documents:
            position = 0
            start = 0
            while start < len(document.text):
                window = document.text[start : start + self._config.chunk_size]
                chunks.append(
                    Chunk(
                        chunk_id=f'{document.doc_id}::{position}',
                        doc_id=document.doc_id,
                        text=window,
                        position=position,
                    ),
                )
                position += 1
                start += step
        return chunks
