"""Reusable job batching.

Both the in-process engine and the RabbitMQ worker accumulate jobs into batches
and flush them through the model's batched inference path. This module extracts
that shared logic so the two transports stay consistent and the batching policy
is defined in one place.
"""
from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class BatchCollector(Generic[T]):
    """Accumulate items into batches of at most ``batch_size``.

    ``collect`` returns a batch as soon as either ``batch_size`` items have been
    gathered or ``batch_timeout_s`` elapsed since the first item arrived. This
    bounds latency under low load while keeping throughput under high load.
    """

    def __init__(self, batch_size: int, batch_timeout_s: float) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if batch_timeout_s < 0:
            raise ValueError("batch_timeout_s must be >= 0")
        self.batch_size = batch_size
        self.batch_timeout_s = batch_timeout_s

    async def collect(self, first: T, get_next: Callable[[float], Awaitable[T | None]]) -> list[T]:
        """Return a batch starting with ``first``.

        ``get_next(timeout_s)`` must resolve to the next item or ``None`` on
        timeout.
        """
        batch = [first]
        deadline = time.monotonic() + self.batch_timeout_s
        while len(batch) < self.batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            item = await get_next(remaining)
            if item is None:
                break
            batch.append(item)
        return batch

    def collect_sync(self, first: T, get_next: Callable[[float], T | None]) -> list[T]:
        """Synchronous variant for thread-based consumers (e.g. the worker)."""
        batch = [first]
        deadline = time.monotonic() + self.batch_timeout_s
        while len(batch) < self.batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            item = get_next(remaining)
            if item is None:
                break
            batch.append(item)
        return batch