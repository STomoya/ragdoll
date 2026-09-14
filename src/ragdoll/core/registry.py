"""Stage registries and chunker-retriever compatibility constraints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar('T')


class InvalidConfigModelError(TypeError):
    """Raised when a stage is registered without a Pydantic config model."""

    def __init__(self, name: str) -> None:
        super().__init__(f"config_model for '{name}' must be a Pydantic BaseModel subclass")


class DuplicateStageNameError(ValueError):
    """Raised when a stage name is registered more than once."""

    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is already registered")


@dataclass
class StageEntry:
    """A registered stage implementation and its config model."""

    cls: type
    config_model: type[BaseModel]


@dataclass
class StageRegistry:
    """Name -> implementation registry for one stage slot (chunker, retriever, ...)."""

    entries: dict[str, StageEntry] = field(default_factory=dict)

    def register(self, name: str, config_model: type[BaseModel]) -> Callable[[type[T]], type[T]]:
        """Register a stage implementation under ``name`` with its Pydantic config model."""
        if not (isinstance(config_model, type) and issubclass(config_model, BaseModel)):
            raise InvalidConfigModelError(name)

        def decorator(cls: type[T]) -> type[T]:
            if name in self.entries:
                raise DuplicateStageNameError(name)
            self.entries[name] = StageEntry(cls=cls, config_model=config_model)
            return cls

        return decorator

    def get(self, name: str) -> StageEntry:
        """Look up a registered stage implementation by name."""
        return self.entries[name]


chunkers = StageRegistry()
query_transforms = StageRegistry()
retrievers = StageRegistry()
rerankers = StageRegistry()
generators = StageRegistry()

# (chunker_name, retriever_name) -> reason, for pairs that must never be combined.
_incompatible: dict[tuple[str, str], str] = {}


def declare_incompatible(chunker_name: str, retriever_name: str, reason: str) -> None:
    """Mark a chunker/retriever pair as incompatible; the runner must skip it."""
    _incompatible[chunker_name, retriever_name] = reason


def is_compatible(chunker_name: str, retriever_name: str) -> bool:
    """Return whether a chunker/retriever pair may be combined."""
    return (chunker_name, retriever_name) not in _incompatible


def incompatibility_reason(chunker_name: str, retriever_name: str) -> str | None:
    """Return why a chunker/retriever pair was declared incompatible, if it was."""
    return _incompatible.get((chunker_name, retriever_name))
