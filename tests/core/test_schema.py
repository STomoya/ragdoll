import pytest
from pydantic import ValidationError

from ragdoll.core.schema import (
    Chunk,
    Document,
    Query,
    RAGResponse,
    ReasoningStep,
    RetrievedContext,
    TransformedQuery,
)


def test_document_defaults():
    doc = Document(doc_id="d1", text="hello")
    assert doc.title is None
    assert doc.metadata == {}


def test_query_defaults():
    query = Query(query_id="q1", text="what?", lang="en")
    assert query.gold_answers == []
    assert query.gold_doc_ids == []


def test_chunk_requires_position():
    with pytest.raises(ValidationError):
        Chunk(chunk_id="c1", doc_id="d1", text="hi")  # ty: ignore[missing-argument]


def test_retrieved_context_score_optional():
    ctx = RetrievedContext(chunk_id="c1", doc_id="d1", text="hi", rank=1)
    assert ctx.score is None


def test_transformed_query_requires_search_texts():
    with pytest.raises(ValidationError):
        TransformedQuery(query_id="q1")  # ty: ignore[missing-argument]


def test_reasoning_step_minimal():
    step = ReasoningStep(step_index=0, action="reformulate_query")
    assert step.input is None
    assert step.output is None


def test_rag_response_defaults():
    response = RAGResponse(
        query_id="q1",
        answer="42",
        retrieved_contexts=[
            RetrievedContext(chunk_id="c1", doc_id="d1", text="hi", rank=1)
        ],
        latency_ms=12.5,
    )
    assert response.citations == []
    assert response.reasoning_trace == []
    assert response.token_usage == {}
