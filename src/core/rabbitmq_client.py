"""RabbitMQ helpers.

RabbitMQ is the message broker decoupling the HTTP layer from the inference
workers. The API publishes a job per masking request; workers consume jobs,
batch them and publish results back through Redis.

A single blocking connection is used per thread. The publisher keeps one
connection guarded by a lock; the worker opens its own connection per thread.
"""
from __future__ import annotations

import json
import threading
from typing import Any

import pika

from .config import settings


class JobPublisher:
    """Thread-safe publisher used by the HTTP layer."""

    def __init__(self, url: str | None = None, queue: str | None = None) -> None:
        self._url = url or settings.rabbitmq_url
        self._queue = queue or settings.queue_name
        self._lock = threading.Lock()
        self._connection: pika.BlockingConnection | None = None
        self._channel: pika.adapters.blocking_connection.BlockingChannel | None = None

    def _ensure(self) -> None:
        if self._connection is None or self._connection.is_closed:
            self._connection = pika.BlockingConnection(pika.URLParameters(self._url))
            self._channel = self._connection.channel()
            self._channel.queue_declare(queue=self._queue, durable=True)

    def publish(self, payload_id: str, payload: str) -> None:
        body = json.dumps({"payload_id": payload_id, "payload": payload}, ensure_ascii=False)
        with self._lock:
            self._ensure()
            self._channel.basic_publish(
                exchange="",
                routing_key=self._queue,
                body=body.encode("utf-8"),
                properties=pika.BasicProperties(delivery_mode=2),
            )

    def close(self) -> None:
        with self._lock:
            if self._connection is not None and not self._connection.is_closed:
                self._connection.close()
            self._connection = None
            self._channel = None


def decode_job(body: bytes) -> dict[str, Any]:
    return json.loads(body.decode("utf-8"))