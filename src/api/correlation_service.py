"""Correlation service for the HTTP layer.

The correlation store is synchronous (memory dict or blocking Redis client). The
HTTP layer is async, so Redis-backed reads must be offloaded to an executor to
avoid blocking the event loop. This service encapsulates that branching so the
endpoint handler stays clean.
"""
from __future__ import annotations

import asyncio
from typing import Any

from ..core.correlation import CorrelationStore, MemoryCorrelationStore, create_correlation_store


class CorrelationService:
    """Async facade over a :class:`CorrelationStore`."""

    def __init__(self, store: CorrelationStore | None = None) -> None:
        self._store = store or create_correlation_store()

    @property
    def store(self) -> CorrelationStore:
        return self._store

    async def get(self, payload_id: str) -> dict[str, Any] | None:
        return await self._run(self._store.get, payload_id)

    async def get_result(self, payload_id: str) -> str | None:
        return await self._run(self._store.get_result, payload_id)

    async def _run(self, fn, *args: Any) -> Any:
        if isinstance(self._store, MemoryCorrelationStore):
            return fn(*args)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, fn, *args)