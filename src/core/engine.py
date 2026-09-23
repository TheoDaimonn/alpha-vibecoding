"""In-process batched inference and masking with a shared correlation store."""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..models.base import Detector
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
        if detector is None:
            from ..models.factory import create_detector

            detector = create_detector()
        self.detector = detector
        self.store = store if store is not None else create_correlation_store()
        self.threads = settings.worker_threads if threads is None else threads
        self.batch_size = settings.worker_batch_size if batch_size is None else batch_size
        self.batch_timeout_s = settings.worker_batch_timeout_s if batch_timeout_s is None else batch_timeout_s
        if self.threads < 1:
            raise ValueError("threads must be >= 1")
        self._collector = BatchCollector(self.batch_size, self.batch_timeout_s)
        self._queue: asyncio.Queue[_Job] = asyncio.Queue(maxsize=settings.worker_queue_size)
        self._workers: list[asyncio.Task] = []
        self._pool: ThreadPoolExecutor | None = None
        self._started = False
        self._pending_chars = 0

    @property
    def running(self) -> bool:
        return self._started and bool(self._workers) and all(not worker.done() for worker in self._workers)

    async def start(self) -> None:
        if self._started:
            return
        self._pool = ThreadPoolExecutor(max_workers=self.threads, thread_name_prefix="pii-inference")
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
        self._started = False
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        close = getattr(self.store, "aclose", None)
        if close is not None:
            await close()
        while not self._queue.empty():
            job = self._queue.get_nowait()
            self._pending_chars -= len(job.payload)
            job.future.cancel()

    async def submit(self, payload_id: str, payload: str, timeout_s: float) -> str | None:
        """Submit a masking job and await the masked result."""
        if not self._started:
            raise RuntimeError("engine is not running")
        validate = getattr(self.detector, "validate_text", None)
        if validate is not None:
            validate(payload)
        if self._pending_chars + len(payload) > settings.worker_queue_chars:
            return None
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()
        try:
            self._queue.put_nowait(_Job(payload_id, payload, fut))
        except asyncio.QueueFull:
            return None
        self._pending_chars += len(payload)
        try:
            return await asyncio.wait_for(fut, timeout=timeout_s)
        except asyncio.TimeoutError:
            fut.cancel()
            return None

    async def _worker_loop(self) -> None:
        """Collect jobs into batches and run the detector."""
        while True:
            first = await self._queue.get()
            batch = [first]
            # Track each collected job even when collection is cancelled.
            async def get_next(timeout_s: float, batch: list[_Job] = batch) -> _Job | None:
                job = await self._get_next(timeout_s)
                if job is not None:
                    batch.append(job)
                return job

            try:
                await self._collector.collect(first, get_next)
                active = [job for job in batch if not job.future.done()]
                if not active:
                    continue
                work = asyncio.get_running_loop().run_in_executor(self._pool, self._process_batch, active)
                try:
                    results = await asyncio.shield(work)
                except asyncio.CancelledError:
                    await asyncio.gather(work, return_exceptions=True)
                    raise
                for job, result in zip(active, results, strict=True):
                    if not job.future.done():
                        if isinstance(result, Exception):
                            job.future.set_exception(result)
                        else:
                            job.future.set_result(result)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - isolate failed batches and keep workers alive
                logger.error("batch processing failed (%s)", type(exc).__name__)
                for job in batch:
                    if not job.future.done():
                        job.future.set_exception(RuntimeError("inference failed"))
            finally:
                self._pending_chars -= sum(len(job.payload) for job in batch)
                for job in batch:
                    if not job.future.done():
                        job.future.cancel()

    async def _get_next(self, timeout_s: float) -> _Job | None:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return None

    def _process_batch(self, batch: list[_Job]) -> list[str | Exception]:
        """Run blocking inference/storage; futures belong to the event loop."""
        results = self.detector.predict_batch([job.payload for job in batch])
        if len(results) != len(batch):
            raise ValueError("detector returned an unexpected number of results")
        masked_results = []
        for job, entities in zip(batch, results, strict=True):
            masked, spans = mask_text(job.payload, entities)
            record = self.store.put_if_absent(
                job.payload_id,
                {"original": job.payload, "masked": masked, "spans": spans_to_dicts(spans)},
            )
            if record["masked"] == job.payload:
                masked_results.append(record["original"])
            else:
                masked_results.append(record["masked"])
        return masked_results
