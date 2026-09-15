"""Single-shot generator: one LLM call over the query and its retrieved contexts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from ragdoll.core.clients import generate_chat
from ragdoll.core.registry import generators
from ragdoll.core.schema import RAGResponse

if TYPE_CHECKING:
    from ragdoll.core.schema import Query, RetrievedContext


class SingleShotGeneratorConfig(BaseModel):
    """Config for SingleShotGenerator."""

    model_name: str = 'gpt-4o-mini'
    max_tokens: int = 512
    temperature: float = 0.0


@generators.register('single_shot', SingleShotGeneratorConfig)
class SingleShotGenerator:
    """Answers a query in a single LLM call, grounded in its retrieved contexts."""

    def __init__(self, config: SingleShotGeneratorConfig) -> None:
        self._config = config

    def __call__(self, query: Query, contexts: list[RetrievedContext]) -> RAGResponse:
        """Generate an answer by calling the LLM once with the query and retrieved contexts."""
        context_block = '\n\n'.join(f'[{i}] {context.text}' for i, context in enumerate(contexts, start=1))
        prompt = (
            f'Answer the question using only the numbered contexts below.\n\n'
            f'Contexts:\n{context_block}\n\nQuestion: {query.text}\nAnswer:'
        )
        answer, latency_ms, token_usage = generate_chat(
            [{'role': 'user', 'content': prompt}],
            self._config.model_name,
            self._config.max_tokens,
            self._config.temperature,
        )
        return RAGResponse(
            query_id=query.query_id,
            answer=answer,
            retrieved_contexts=contexts,
            latency_ms=latency_ms,
            token_usage=token_usage,
        )
