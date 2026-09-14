"""Core data model shared by every stage and metric; see AGENTS.md's Data model section."""

from typing import Any

from pydantic import BaseModel

IndexHandle = object
# opaque: intentionally not `Any`, so no stage can assume anything about its structure.


class Document(BaseModel):
    """A single source document, before chunking."""

    doc_id: str
    text: str
    title: str | None = None
    metadata: dict[str, Any] = {}


class Query(BaseModel):
    """A benchmark query, with gold references for scoring."""

    query_id: str
    text: str
    lang: str
    gold_answers: list[str] = []
    gold_doc_ids: list[str] = []
    metadata: dict[str, Any] = {}


class Chunk(BaseModel):
    """A retrieval unit produced by a chunker from one document."""

    chunk_id: str
    doc_id: str
    text: str
    position: int
    metadata: dict[str, Any] = {}


class RetrievedContext(BaseModel):
    """One chunk returned by a retriever for a given query, with its rank."""

    chunk_id: str
    doc_id: str
    text: str
    score: float | None = None
    rank: int


class TransformedQuery(BaseModel):
    """Output of the query-transform stage: one or more search strings."""

    query_id: str
    search_texts: list[str]
    metadata: dict[str, Any] = {}


class ReasoningStep(BaseModel):
    """One internal action taken by a multi-hop/agentic generator."""

    step_index: int
    action: str
    input: str | None = None
    output: str | None = None


class RAGResponse(BaseModel):
    """A generator's final answer plus everything it was conditioned on."""

    query_id: str
    answer: str
    retrieved_contexts: list[RetrievedContext]
    citations: list[tuple[int, str]] = []
    reasoning_trace: list[ReasoningStep] = []
    latency_ms: float
    token_usage: dict[str, int] = {}
