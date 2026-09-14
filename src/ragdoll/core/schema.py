from typing import Any

from pydantic import BaseModel


class Document(BaseModel):
    doc_id: str
    text: str
    title: str | None = None
    metadata: dict[str, Any] = {}


class Query(BaseModel):
    query_id: str
    text: str
    lang: str
    gold_answers: list[str] = []
    gold_doc_ids: list[str] = []
    metadata: dict[str, Any] = {}


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    position: int
    metadata: dict[str, Any] = {}


class RetrievedContext(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float | None = None
    rank: int


class TransformedQuery(BaseModel):
    query_id: str
    search_texts: list[str]
    metadata: dict[str, Any] = {}


class ReasoningStep(BaseModel):
    step_index: int
    action: str
    input: str | None = None
    output: str | None = None


class RAGResponse(BaseModel):
    query_id: str
    answer: str
    retrieved_contexts: list[RetrievedContext]
    citations: list[tuple[int, str]] = []
    reasoning_trace: list[ReasoningStep] = []
    latency_ms: float
    token_usage: dict[str, int] = {}
