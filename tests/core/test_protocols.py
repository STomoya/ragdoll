"""Tests for core.protocols."""

from __future__ import annotations

from ragdoll.core.protocols import (
    ChunkerProtocol,
    GeneratorProtocol,
    QueryTransformProtocol,
    RerankerProtocol,
    RetrieverProtocol,
)


class _Callable:
    def __call__(self, *_args: object) -> None:
        return None


class _Retriever:
    def build_index(self, _chunks: object) -> None:
        return None

    def retrieve(self, _transformed_query: object, _index_handle: object) -> None:
        return None


def test_callable_satisfies_single_method_protocols() -> None:
    instance = _Callable()
    assert isinstance(instance, ChunkerProtocol)
    assert isinstance(instance, QueryTransformProtocol)
    assert isinstance(instance, RerankerProtocol)
    assert isinstance(instance, GeneratorProtocol)


def test_plain_object_satisfies_no_protocol() -> None:
    instance = object()
    assert not isinstance(instance, ChunkerProtocol)
    assert not isinstance(instance, RetrieverProtocol)


def test_retriever_protocol_needs_both_methods() -> None:
    assert isinstance(_Retriever(), RetrieverProtocol)
    assert not isinstance(_Callable(), RetrieverProtocol)
