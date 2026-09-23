"""Redis correlation and result storage for the inference service."""
from __future__ import annotations

import json
from typing import Any

import redis

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
