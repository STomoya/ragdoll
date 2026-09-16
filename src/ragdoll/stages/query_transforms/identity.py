"""Identity query transform: passes the query text through unchanged."""

from __future__ import annotations

from pydantic import BaseModel

from ragdoll.core.registry import query_transforms
from ragdoll.core.schema import Query, TransformedQuery


class IdentityQueryTransformConfig(BaseModel):
    """Config for IdentityQueryTransform (no parameters)."""


@query_transforms.register('identity', IdentityQueryTransformConfig)
class IdentityQueryTransform:
    """Passes the query text through as the single search text."""

    def __init__(self, config: IdentityQueryTransformConfig) -> None:
        self._config = config

    def __call__(self, query: Query) -> TransformedQuery:
        """Transform a query by passing its text through as the single search text."""
        return TransformedQuery(query_id=query.query_id, search_texts=[query.text])
