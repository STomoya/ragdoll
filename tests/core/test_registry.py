import pytest
from pydantic import BaseModel

from ragdoll.core.registry import (
    StageRegistry,
    declare_incompatible,
    incompatibility_reason,
    is_compatible,
)


class DummyConfig(BaseModel):
    value: int = 0


def test_register_and_get():
    registry = StageRegistry()

    @registry.register("dummy", DummyConfig)
    class DummyStage:
        pass

    entry = registry.get("dummy")
    assert entry.cls is DummyStage
    assert entry.config_model is DummyConfig


def test_register_rejects_bare_dict_config():
    registry = StageRegistry()
    with pytest.raises(TypeError):
        registry.register("dummy", dict)  # ty: ignore[invalid-argument-type]


def test_register_rejects_duplicate_name():
    registry = StageRegistry()
    registry.register("dummy", DummyConfig)(object)
    with pytest.raises(ValueError):
        registry.register("dummy", DummyConfig)(object)


def test_compatibility_defaults_to_true():
    assert is_compatible("fixed_size", "dense") is True
    assert incompatibility_reason("fixed_size", "dense") is None


def test_declare_incompatible():
    declare_incompatible(
        "graph_chunker", "bm25", "graph chunks require a graph-aware retriever"
    )
    assert is_compatible("graph_chunker", "bm25") is False
    assert (
        incompatibility_reason("graph_chunker", "bm25")
        == "graph chunks require a graph-aware retriever"
    )
