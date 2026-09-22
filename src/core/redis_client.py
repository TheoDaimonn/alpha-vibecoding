"""Thin Redis helpers.

Redis is used for two things:
  1. The payload_id correlation store (masking -> unmasking). For every
     payload_id we keep the original text, the masked text and the spans so the
     second request with the same id can restore the original.
  2. The result store + notification channel so the HTTP layer can wait for a
     worker to finish a job without holding a RabbitMQ connection open.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import redis
import redis.asyncio as aioredis

from .config import settings


class RedisStore:
    def __init__(self, url: str | None = None, ttl_s: int | None = None) -> None:
        self._client = redis.Redis.from_url(url or settings.redis_url, decode_responses=True)
        self._ttl = ttl_s or settings.result_ttl_s

    # --- correlation store ---
    def get_correlation(self, payload_id: str) -> dict[str, Any] | None:
        raw = self._client.get(f"corr:{payload_id}")
        if raw is None:
            return None
        return json.loads(raw)

    def set_correlation(self, payload_id: str, data: dict[str, Any]) -> None:
        self._client.set(f"corr:{payload_id}", json.dumps(data, ensure_ascii=False), ex=self._ttl)

    # --- result store ---
    def get_result(self, payload_id: str) -> str | None:
        return self._client.get(f"result:{payload_id}")

    def set_result(self, payload_id: str, result: str) -> None:
        self._client.set(f"result:{payload_id}", result, ex=self._ttl)

    def publish_result(self, payload_id: str, result: str) -> None:
        """Publish a result notification (used by sync workers)."""
        self._client.publish(
            ResultNotifier.CHANNEL,
            json.dumps({"payload_id": payload_id, "result": result}, ensure_ascii=False),
        )

    def wait_for_result(self, payload_id: str, timeout_s: float | None = None, poll_s: float | None = None) -> str | None:
        """Poll the result key until it appears or the timeout elapses."""
        deadline = time.monotonic() + (timeout_s or settings.request_timeout_s)
        interval = poll_s or settings.poll_interval_s
        while time.monotonic() < deadline:
            value = self.get_result(payload_id)
            if value is not None:
                return value
            time.sleep(interval)
        return None

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except redis.RedisError:
            return False


class AsyncRedisStore:
    """Async Redis helpers for the HTTP layer (high concurrency)."""

    def __init__(self, url: str | None = None, ttl_s: int | None = None, max_connections: int = 2000) -> None:
        pool = aioredis.ConnectionPool.from_url(
            url or settings.redis_url, decode_responses=True, max_connections=max_connections
        )
        self._client = aioredis.Redis(connection_pool=pool)
        self._ttl = ttl_s or settings.result_ttl_s

    async def get_correlation(self, payload_id: str) -> dict[str, Any] | None:
        raw = await self._client.get(f"corr:{payload_id}")
        if raw is None:
            return None
        return json.loads(raw)

    async def get_result(self, payload_id: str) -> str | None:
        return await self._client.get(f"result:{payload_id}")

    async def wait_for_result(self, payload_id: str, timeout_s: float | None = None, poll_s: float | None = None) -> str | None:
        import asyncio

        deadline = time.monotonic() + (timeout_s or settings.request_timeout_s)
        interval = poll_s or settings.poll_interval_s
        while time.monotonic() < deadline:
            value = await self.get_result(payload_id)
            if value is not None:
                return value
            await asyncio.sleep(interval)
        return None

    async def ping(self) -> bool:
        try:
            return bool(await self._client.ping())
        except aioredis.RedisError:
            return False


class ResultNotifier:
    """Redis pub/sub result notification for the HTTP layer.

    A single background subscriber listens on the shared ``results`` channel.
    Workers publish ``{payload_id, result}`` messages; the notifier resolves the
    matching :class:`asyncio.Future` so API requests wake up without polling.
    """

    CHANNEL = "pii:results"

    def __init__(self, url: str | None = None) -> None:
        self._url = url or settings.redis_url
        self._pub = aioredis.Redis.from_url(self._url, decode_responses=True)
        self._sub = aioredis.Redis.from_url(self._url, decode_responses=True)
        self._futures: dict[str, asyncio.Future[str]] = {}
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._listen())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _listen(self) -> None:
        pubsub = self._sub.pubsub()
        await pubsub.subscribe(self.CHANNEL)
        try:
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue
                data = json.loads(message["data"])
                pid = data["payload_id"]
                async with self._lock:
                    fut = self._futures.pop(pid, None)
                if fut is not None and not fut.done():
                    fut.set_result(data["result"])
        finally:
            await pubsub.unsubscribe(self.CHANNEL)

    async def wait(self, payload_id: str, timeout_s: float) -> str | None:
        fut: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        async with self._lock:
            self._futures[payload_id] = fut
        try:
            return await asyncio.wait_for(fut, timeout=timeout_s)
        except asyncio.TimeoutError:
            async with self._lock:
                self._futures.pop(payload_id, None)
            return None

    async def publish(self, payload_id: str, result: str) -> None:
        await self._pub.publish(self.CHANNEL, json.dumps({"payload_id": payload_id, "result": result}, ensure_ascii=False))