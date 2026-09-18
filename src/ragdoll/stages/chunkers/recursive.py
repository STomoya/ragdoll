"""Recursive text splitter: splits on a separator hierarchy, then packs pieces into chunk_size windows."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from ragdoll.core.registry import chunkers
from ragdoll.core.schema import Chunk, Document

DEFAULT_SEPARATORS = ('\n\n', '\n', '. ', ' ', '')


class RecursiveTextSplitterConfig(BaseModel):
    """Config for RecursiveTextSplitter."""

    chunk_size: int = Field(default=500, gt=0)
    overlap: int = Field(default=50, ge=0)
    separators: tuple[str, ...] = Field(default=DEFAULT_SEPARATORS)

    @model_validator(mode='after')
    def _check_overlap(self) -> RecursiveTextSplitterConfig:
        if self.overlap >= self.chunk_size:
            msg = 'overlap must be smaller than chunk_size'
            raise ValueError(msg)
        return self


def _split_spans(text: str, separators: tuple[str, ...], chunk_size: int, offset: int) -> list[tuple[int, int]]:
    """Recursively split text on the first separator, descending into any piece still over chunk_size.

    Returns absolute (start, end) spans into the original document text, so
    merging can later re-slice it instead of rebuilding text from pieces
    (which would lose the original separators/whitespace).
    """
    separator, *rest = separators
    parts = text.split(separator) if separator else list(text)
    spans = []
    cursor = offset
    for part in parts:
        if part:
            start, end = cursor, cursor + len(part)
            if len(part) <= chunk_size or not rest:
                spans.append((start, end))
            else:
                spans.extend(_split_spans(part, tuple(rest), chunk_size, start))
        cursor += len(part) + len(separator)
    return spans


def _merge_spans(spans: list[tuple[int, int]], chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    """Greedily pack adjacent spans into windows up to chunk_size, overlapping each window's tail into the next."""
    if not spans:
        return []
    windows = []
    start, end = spans[0]
    for piece_start, piece_end in spans[1:]:
        if piece_end - start <= chunk_size:
            end = piece_end
        else:
            windows.append((start, end))
            start = max(end - overlap, 0) if overlap else piece_start
            end = piece_end
    windows.append((start, end))
    return windows


@chunkers.register('recursive', RecursiveTextSplitterConfig)
class RecursiveTextSplitter:
    """Splits each document on a separator hierarchy (paragraphs, lines, sentences, words, chars), then packs
    the resulting pieces into ~chunk_size windows with overlap.
    """

    def __init__(self, config: RecursiveTextSplitterConfig) -> None:
        self._config = config

    def __call__(self, documents: list[Document]) -> list[Chunk]:
        """Split documents into recursively-derived, size-packed chunks."""
        chunks: list[Chunk] = []
        for document in documents:
            spans = _split_spans(document.text, self._config.separators, self._config.chunk_size, 0)
            windows = _merge_spans(spans, self._config.chunk_size, self._config.overlap)
            for position, (start, end) in enumerate(windows):
                chunks.append(
                    Chunk(
                        chunk_id=f'{document.doc_id}::{position}',
                        doc_id=document.doc_id,
                        text=document.text[start:end],
                        position=position,
                    ),
                )
        return chunks
