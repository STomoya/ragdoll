"""Reranker implementations: (Query, list[RetrievedContext]) -> list[RetrievedContext]."""

from ragdoll.stages.rerankers.identity import IdentityReranker, IdentityRerankerConfig

__all__ = ['IdentityReranker', 'IdentityRerankerConfig']
