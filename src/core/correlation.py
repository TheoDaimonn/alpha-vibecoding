"""Correlation store: payload_id -> {original, masked, spans}.

The store is used to correlate the masking step with the unmasking step for the
same payload_id. Two backends are provided:

  * ``memory`` (default) — an in-process dict. Fastest; suitable for a
    single-process deployment. Not shared across API workers.
  * ``redis`` — Redis-backed; shared across processes/nodes.

The engine writes correlations; the HTTP layer reads them. Both backends
implement the :class:`CorrelationStore` protocol so callers never depend on a
concrete backend.
"""
from __future__ import annotations

import threading
from typing import Any, Protocol, runtime_checkable

from .config import settings


@runtime_checkable
class CorrelationStore(Protocol):
    """Storage for masking/unmasking correlation records.

    A record maps a ``payload_id`` to ``{"original", "masked", "spans"}``.
    Implementations must be thread-safe.
    """

    def get(self, payload_id: str) -> dict[str, Any] | None: ...

    def set(self, payload_id: str, data: dict[str, Any]) -> None: ...

    def get_result(self, payload_id: str) -> str | None: ...

    def set_result(self, payload_id: str, result: str) -> None: ...


class MemoryCorrelationStore:
    """In-process correlation store (fastest, single process)."""

    def __init__(self, ttl_s: int | None = None) -> None:
        self._data: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_s or settings.result_ttl_s

    def get(self, payload_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._data.get(payload_id)

    def set(self, payload_id: str, data: dict[str, Any]) -> None:
        with self._lock:
            self._data[payload_id] = data

    def get_result(self, payload_id: str) -> str | None:
        with self._lock:
            row = self._data.get(payload_id)
            return row.get("masked") if row else None

    def set_result(self, payload_id: str, result: str) -> None:
        with self._lock:
            row = self._data.setdefault(payload_id, {})
            row["masked"] = result


class RedisCorrelationStore:
    """Redis-backed correlation store (shared across processes)."""

    def __init__(self, url: str | None = None, ttl_s: int | None = None) -> None:
        from .redis_client import RedisStore

        self._store = RedisStore(url=url, ttl_s=ttl_s)

    def get(self, payload_id: str) -> dict[str, Any] | None:
        return self._store.get_correlation(payload_id)

    def set(self, payload_id: str, data: dict[str, Any]) -> None:
        self._store.set_correlation(payload_id, data)

    def get_result(self, payload_id: str) -> str | None:
        return self._store.get_result(payload_id)

    def set_result(self, payload_id: str, result: str) -> None:
        self._store.set_result(payload_id, result)


def create_correlation_store(backend: str | None = None) -> CorrelationStore:
    """Build the configured correlation store backend."""
    kind = (backend or settings.correlation_store).lower()
    if kind == "redis":
        return RedisCorrelationStore()
    return MemoryCorrelationStore()