"""Redis correlation and result storage for the inference service."""
from __future__ import annotations

import json
from typing import Any

import redis
import redis.asyncio as async_redis

from .config import settings


class RedisStore:
    def __init__(self, url: str | None = None, ttl_s: int | None = None) -> None:
        self._client = redis.Redis.from_url(url or settings.redis_url, decode_responses=True,
            socket_connect_timeout=2, socket_timeout=2,
       )
        self._async_client = async_redis.Redis.from_url(
            url or settings.redis_url, decode_responses=True,
            socket_connect_timeout=2, socket_timeout=2, max_connections=128,
        )
        self._ttl = settings.result_ttl_s if ttl_s is None else ttl_s
        if self._ttl <= 0:
            raise ValueError("ttl_s must be positive")

    def ping(self) -> bool:
        return bool(self._client.ping())

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


    def put_if_absent(self, payload_id: str, data: dict[str, Any]) -> dict[str, Any]:
        # One atomic operation: a competing worker cannot replace the original.
        script = """
        local existing = redis.call('GET', KEYS[1])
        if existing then return existing end
        redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
        return ARGV[1]
        """
        raw = self._client.eval(
            script, 1, f"corr:{payload_id}",
            json.dumps(data, ensure_ascii=False), self._ttl,
        )
        return json.loads(raw)


    async def aget(self, payload_id: str) -> dict[str, Any] | None:
        raw = await self._async_client.get(f"corr:{payload_id}")
        return json.loads(raw) if raw is not None else None

    async def aping(self) -> bool:
        return bool(await self._async_client.ping())

    async def aclose(self) -> None:
        await self._async_client.aclose()
        self._client.close()
