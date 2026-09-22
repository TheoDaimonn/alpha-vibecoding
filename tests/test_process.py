"""Functional tests for the /process contract using in-memory fakes.

These tests do not load the real model or require Redis/RabbitMQ. They exercise
the masking/unmasking round-trip and the idempotency rules of the endpoint.
"""
from __future__ import annotations

import threading
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api import main as api_main
from src.core import masking
from src.core.engine import InferenceEngine
from ru_pii.schema import Entity


class FakeReplica:
    """Deterministic model double: detects a fixed set of entities."""

    name = "fake"

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def predict(self, text: str) -> list[Entity]:
        with self._lock:
            return self._detect(text)

    def predict_batch(self, texts: list[str]) -> list[list[Entity]]:
        with self._lock:
            return [self._detect(t) for t in texts]

    @staticmethod
    def _detect(text: str) -> list[Entity]:
        out = []
        for literal, label in [
            ("Иванов Иван Иванович", "PERSON"),
            ("test@example.org", "EMAIL"),
            ("4509 123456", "PASSPORT_NUMBER"),
            ("+7 900 123-45-67", "PHONE"),
        ]:
            idx = text.find(literal)
            if idx >= 0:
                out.append(Entity(idx, idx + len(literal), label, 1.0, literal))
        return out


class FakeRedis:
    def __init__(self) -> None:
        self._corr: dict[str, dict[str, Any]] = {}
        self._result: dict[str, str] = {}

    def get(self, pid: str) -> dict[str, Any] | None:
        return self._corr.get(pid)

    def set(self, pid: str, data: dict[str, Any]) -> None:
        self._corr[pid] = data

    def get_result(self, pid: str) -> str | None:
        return self._result.get(pid)

    def set_result(self, pid: str, result: str) -> None:
        self._result[pid] = result


@pytest.fixture()
def client() -> TestClient:
    replica = FakeReplica()
    shared_redis = FakeRedis()
    engine = InferenceEngine(detector=replica, store=shared_redis, threads=1, batch_size=4, batch_timeout_s=0.01)
    api_main._store = shared_redis
    api_main._engine = engine
    with TestClient(api_main.app) as c:
        yield c
    api_main._store = None
    api_main._engine = None


def test_mask_then_unmask_roundtrip(client: TestClient) -> None:
    text = "Клиент Иванов Иван Иванович, email: test@example.org, паспорт 4509 123456"
    pid = "roundtrip-1"

    r1 = client.post("/process", json={"payload": text, "payload_id": pid})
    assert r1.status_code == 200
    masked = r1.json()["result"]
    assert "Иванов Иван Иванович" not in masked
    assert "test@example.org" not in masked

    r2 = client.post("/process", json={"payload": masked, "payload_id": pid})
    assert r2.status_code == 200
    assert r2.json()["result"] == text


def test_masking_retry_is_idempotent(client: TestClient) -> None:
    text = "Клиент Иванов Иван Иванович, email: test@example.org"
    pid = "retry-1"

    r1 = client.post("/process", json={"payload": text, "payload_id": pid})
    masked = r1.json()["result"]

    r2 = client.post("/process", json={"payload": text, "payload_id": pid})
    assert r2.status_code == 200
    assert r2.json()["result"] == masked


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_masking_preserves_positions(client: TestClient) -> None:
    text = "Иванов Иван Иванович test@example.org"
    pid = "pos-1"
    r = client.post("/process", json={"payload": text, "payload_id": pid})
    masked = r.json()["result"]
    assert len(masked) == len(text)
    assert masked[:6] == "Иванов" or masked[:6] == "******"