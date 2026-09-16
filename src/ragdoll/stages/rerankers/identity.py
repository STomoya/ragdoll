"""Identity reranker: returns the retrieved contexts unchanged."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from ragdoll.core.registry import rerankers

if TYPE_CHECKING:
    from ragdoll.core.schema import Query, RetrievedContext


class IdentityRerankerConfig(BaseModel):
    """Config for IdentityReranker (no parameters)."""


@rerankers.register('identity', IdentityRerankerConfig)
class IdentityReranker:
    """Passes retrieved contexts through unchanged."""

    def __init__(self, config: IdentityRerankerConfig) -> None:
        self._config = config

    def __call__(self, _query: Query, contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        """Return contexts unchanged."""
        return contexts
