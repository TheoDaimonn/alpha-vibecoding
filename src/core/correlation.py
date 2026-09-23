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
from collections import OrderedDict
from copy import deepcopy
from time import monotonic
from typing import Any, Protocol, runtime_checkable

from .config import settings


@runtime_checkable
class CorrelationStore(Protocol):
    """Storage for masking/unmasking correlation records.

    A record maps a ``payload_id`` to ``{"original", "masked", "spans"}``.
    Implementations must be thread-safe.
    """

    def ping(self) -> bool: ...

    def get(self, payload_id: str) -> dict[str, Any] | None: ...

    def set(self, payload_id: str, data: dict[str, Any]) -> None: ...

    def put_if_absent(self, payload_id: str, data: dict[str, Any]) -> dict[str, Any]: ...

    def get_result(self, payload_id: str) -> str | None: ...

    def set_result(self, payload_id: str, result: str) -> None: ...


class MemoryCorrelationStore:
    """In-process correlation store (fastest, single process)."""

    def __init__(self, ttl_s: int | None = None) -> None:
        self._data: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._lock = threading.Lock()
        self._ttl = settings.result_ttl_s if ttl_s is None else ttl_s
        if self._ttl <= 0:
            raise ValueError("ttl_s must be positive")

    def ping(self) -> bool:
        return True

    def _purge_expired(self) -> None:
        now = monotonic()
        while self._data:
            deadline, _ = next(iter(self._data.values()))
            if deadline > now:
                break
            self._data.popitem(last=False)

    def _write(self, payload_id: str, data: dict[str, Any]) -> None:
        self._data[payload_id] = (monotonic() + self._ttl, deepcopy(data))
        self._data.move_to_end(payload_id)

    def get(self, payload_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._purge_expired()
            row = self._data.get(payload_id)
            return deepcopy(row[1]) if row is not None else None

    def set(self, payload_id: str, data: dict[str, Any]) -> None:
        with self._lock:
            self._purge_expired()
            self._write(payload_id, data)

    def put_if_absent(self, payload_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Atomically preserve the first original and return the winning record."""
        with self._lock:
            self._purge_expired()
            if payload_id not in self._data:
                self._write(payload_id, data)
            return deepcopy(self._data[payload_id][1])

    def get_result(self, payload_id: str) -> str | None:
        row = self.get(payload_id)
        return row.get("masked") if row else None

    def set_result(self, payload_id: str, result: str) -> None:
        with self._lock:
            self._purge_expired()
            existing = self._data.get(payload_id)
            row = dict(existing[1]) if existing else {}
            row["masked"] = result
            self._write(payload_id, row)


class RedisCorrelationStore:
    """Redis-backed correlation store (shared across processes)."""

    def __init__(self, url: str | None = None, ttl_s: int | None = None) -> None:
        from .redis_client import RedisStore

        self._store = RedisStore(url=url, ttl_s=ttl_s)

    async def aget(self, payload_id: str) -> dict[str, Any] | None:
        return await self._store.aget(payload_id)

    async def aping(self) -> bool:
        return await self._store.aping()

    async def aclose(self) -> None:
        await self._store.aclose()

    def ping(self) -> bool:
        return self._store.ping()

    def get(self, payload_id: str) -> dict[str, Any] | None:
        return self._store.get_correlation(payload_id)

    def set(self, payload_id: str, data: dict[str, Any]) -> None:
        self._store.set_correlation(payload_id, data)

    def put_if_absent(self, payload_id: str, data: dict[str, Any]) -> dict[str, Any]:
        return self._store.put_if_absent(payload_id, data)

    def get_result(self, payload_id: str) -> str | None:
        return self._store.get_result(payload_id)

    def set_result(self, payload_id: str, result: str) -> None:
        self._store.set_result(payload_id, result)


def create_correlation_store(backend: str | None = None) -> CorrelationStore:
    """Build the configured correlation store backend."""
    kind = (backend or settings.correlation_store).lower()
    if kind == "redis":
        return RedisCorrelationStore()
    if kind == "memory":
        return MemoryCorrelationStore()
    raise ValueError("unknown correlation store backend")
