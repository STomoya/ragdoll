from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T")


@dataclass
class StageEntry:
    cls: type
    config_model: type[BaseModel]


@dataclass
class StageRegistry:
    entries: dict[str, StageEntry] = field(default_factory=dict)

    def register(
        self, name: str, config_model: type[BaseModel]
    ) -> Callable[[type[T]], type[T]]:
        if not (isinstance(config_model, type) and issubclass(config_model, BaseModel)):
            raise TypeError(
                f"config_model for '{name}' must be a Pydantic BaseModel subclass"
            )

        def decorator(cls: type[T]) -> type[T]:
            if name in self.entries:
                raise ValueError(f"'{name}' is already registered")
            self.entries[name] = StageEntry(cls=cls, config_model=config_model)
            return cls

        return decorator

    def get(self, name: str) -> StageEntry:
        return self.entries[name]


chunkers = StageRegistry()
query_transforms = StageRegistry()
retrievers = StageRegistry()
rerankers = StageRegistry()
generators = StageRegistry()

# (chunker_name, retriever_name) -> reason, for pairs that must never be combined.
_incompatible: dict[tuple[str, str], str] = {}


def declare_incompatible(chunker_name: str, retriever_name: str, reason: str) -> None:
    _incompatible[(chunker_name, retriever_name)] = reason


def is_compatible(chunker_name: str, retriever_name: str) -> bool:
    return (chunker_name, retriever_name) not in _incompatible


def incompatibility_reason(chunker_name: str, retriever_name: str) -> str | None:
    return _incompatible.get((chunker_name, retriever_name))
