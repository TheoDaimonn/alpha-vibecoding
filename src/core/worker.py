"""Inference worker.

A worker process owns one loaded model replica and runs several consumer
threads. Each thread pulls jobs from RabbitMQ, accumulates them into a local
batch and flushes the batch through the model's batched inference path. This
gives two levels of parallelism:

  * multiple worker processes (replicas) each with its own model instance;
  * multiple threads per worker, each batching many texts per model call.

For every masking job the worker:
  1. runs the model to detect PII entities;
  2. builds a reversible mask (positions preserved);
  3. stores the correlation (original + masked + spans) in Redis;
  4. writes the masked result to Redis so the HTTP layer can return it.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Any

import pika

from ..models.base import Detector
from ..models.factory import create_detector
from .batcher import BatchCollector
from .config import settings
from .masking import mask_text, spans_to_dicts
from .redis_client import RedisStore

logger = logging.getLogger("pii.worker")


class Worker:
    def __init__(
        self,
        *,
        detector: Detector | None = None,
        redis: RedisStore | None = None,
        url: str | None = None,
        queue_name: str | None = None,
        threads: int | None = None,
        batch_size: int | None = None,
        batch_timeout_s: float | None = None,
    ) -> None:
        self.detector = detector or create_detector()
        self.redis = redis or RedisStore()
        self.url = url or settings.rabbitmq_url
        self.queue_name = queue_name or settings.queue_name
        self.threads = threads or settings.worker_threads
        self.batch_size = batch_size or settings.worker_batch_size
        self.batch_timeout_s = batch_timeout_s or settings.worker_batch_timeout_s
        self._collector = BatchCollector(self.batch_size, self.batch_timeout_s)
        self._jobs: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=100_000)
        self._stop = threading.Event()

    # --- job processing ---
    def _process_one(self, job: dict[str, Any]) -> None:
        payload_id = str(job["payload_id"])
        payload = str(job["payload"])
        entities = self.detector.predict(payload)
        self._store_result(payload_id, payload, entities)

    def _store_result(self, payload_id: str, payload: str, entities: list[Any]) -> None:
        masked, spans = mask_text(payload, entities)
        self.redis.set_correlation(
            payload_id,
            {"original": payload, "masked": masked, "spans": spans_to_dicts(spans)},
        )
        self.redis.set_result(payload_id, masked)
        self.redis.publish_result(payload_id, masked)

    def _process_batch(self, jobs: list[dict[str, Any]]) -> None:
        texts = [str(j["payload"]) for j in jobs]
        ids = [str(j["payload_id"]) for j in jobs]
        results = self.detector.predict_batch(texts)
        for payload_id, payload, entities in zip(ids, texts, results):
            self._store_result(payload_id, payload, entities)

    def _drain(self) -> None:
        """Pull jobs from the queue, batch them and run inference."""
        while not self._stop.is_set():
            try:
                first = self._jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            batch = self._collector.collect_sync(first, self._get_next)
            try:
                if len(batch) == 1:
                    self._process_one(batch[0])
                else:
                    self._process_batch(batch)
            except Exception:  # noqa: BLE001 - keep the worker alive
                logger.exception("failed to process batch of %d jobs", len(batch))

    def _get_next(self, timeout_s: float) -> dict[str, Any] | None:
        try:
            return self._jobs.get(timeout=timeout_s)
        except queue.Empty:
            return None

    # --- rabbitmq consumption ---
    def _consume(self) -> None:
        connection = pika.BlockingConnection(pika.URLParameters(self.url))
        channel = connection.channel()
        channel.queue_declare(queue=self.queue_name, durable=True)
        channel.basic_qos(prefetch_count=self.batch_size * 2)

        def on_message(_ch, _method, _properties, body) -> None:  # type: ignore[no-untyped-def]
            try:
                job = json.loads(body.decode("utf-8"))
                self._jobs.put(job)
            except Exception:  # noqa: BLE001
                logger.exception("dropping malformed job")
            finally:
                _ch.basic_ack(delivery_tag=_method.delivery_tag)

        channel.basic_consume(queue=self.queue_name, on_message_callback=on_message)
        try:
            channel.start_consuming()
        finally:
            connection.close()

    def run(self) -> None:
        threads = [threading.Thread(target=self._drain, daemon=True) for _ in range(self.threads)]
        for t in threads:
            t.start()
        logger.info(
            "worker started: %d threads, batch=%d timeout=%.3fs",
            self.threads,
            self.batch_size,
            self.batch_timeout_s,
        )
        self._consume()

    def stop(self) -> None:
        self._stop.set()