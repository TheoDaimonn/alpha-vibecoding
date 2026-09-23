"""In-process batched inference and masking with a shared correlation store."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from ..models.base import Detector
from ..models.factory import create_detector
from .batcher import BatchCollector
from .config import settings
from .correlation import CorrelationStore, create_correlation_store
from .masking import mask_text, spans_to_dicts

logger = logging.getLogger("pii.engine")


@dataclass(slots=True)
class _Job:
    payload_id: str
    payload: str
    future: asyncio.Future[str]


class InferenceEngine:
    """Async engine: submit a masking job and await its result."""

    def __init__(
        self,
        *,
        detector: Detector | None = None,
        store: CorrelationStore | None = None,
        threads: int | None = None,
        batch_size: int | None = None,
        batch_timeout_s: float | None = None,
    ) -> None:
        self.detector = detector or create_detector()
        self.store = store or create_correlation_store()
        self.threads = threads or settings.worker_threads
        self.batch_size = batch_size or settings.worker_batch_size
        self.batch_timeout_s = batch_timeout_s or settings.worker_batch_timeout_s
        self._collector = BatchCollector(self.batch_size, self.batch_timeout_s)
        self._queue: asyncio.Queue[_Job] = asyncio.Queue(maxsize=100_000)
        self._workers: list[asyncio.Task] = []
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        loop = asyncio.get_running_loop()
        self._workers = [loop.create_task(self._worker_loop()) for _ in range(self.threads)]
        logger.info(
            "engine started: %d threads, batch=%d timeout=%.3fs",
            self.threads,
            self.batch_size,
            self.batch_timeout_s,
        )

    async def stop(self) -> None:
        for w in self._workers:
            w.cancel()
        self._workers = []

    async def submit(self, payload_id: str, payload: str, timeout_s: float) -> str | None:
        """Submit a masking job and await the masked result."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()
        await self._queue.put(_Job(payload_id, payload, fut))
        try:
            return await asyncio.wait_for(fut, timeout=timeout_s)
        except asyncio.TimeoutError:
            fut.cancel()
            return None

    async def _worker_loop(self) -> None:
        """Collect jobs into batches and run the detector."""
        while True:
            first = await self._queue.get()
            batch = await self._collector.collect(first, self._get_next)
            await asyncio.get_running_loop().run_in_executor(None, self._process_batch, batch)

    async def _get_next(self, timeout_s: float) -> _Job | None:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return None

    def _process_batch(self, batch: list[_Job]) -> None:
        texts = [j.payload for j in batch]
        try:
            results = self.detector.predict_batch(texts)
            for job, entities in zip(batch, results):
                masked, spans = mask_text(job.payload, entities)
                self.store.set(
                    job.payload_id,
                    {"original": job.payload, "masked": masked, "spans": spans_to_dicts(spans)},
                )
                self.store.set_result(job.payload_id, masked)
                if not job.future.done():
                    job.future.set_result(masked)
        except Exception:  # noqa: BLE001 - keep the worker alive
            logger.exception("batch processing failed")
            for job in batch:
                if not job.future.done():
                    job.future.set_exception(RuntimeError("inference failed"))
