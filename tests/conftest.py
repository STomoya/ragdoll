"""Shared pytest fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

from ragdoll.core.clients import _get_openai_client, _load_embedding_model


@pytest.fixture(autouse=True)
def _clear_client_caches() -> Generator[None, None, None]:
    yield
    _load_embedding_model.cache_clear()
    _get_openai_client.cache_clear()
