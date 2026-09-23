import asyncio
import threading
import unittest

from src.core.correlation import MemoryCorrelationStore
from src.core.engine import InferenceEngine


class EmptyDetector:
    def predict_batch(self, texts):
        return [[] for _ in texts]


class BrokenDetector:
    def predict_batch(self, texts):
        return []


class EngineTests(unittest.IsolatedAsyncioTestCase):
    def engine(self, detector=None, **kwargs):
        return InferenceEngine(
            detector=detector if detector is not None else EmptyDetector(),
            store=MemoryCorrelationStore(), threads=1, **kwargs,
        )

    async def test_batch_results_and_restart(self):
        engine = self.engine(batch_size=2, batch_timeout_s=0.01)
        for _ in range(2):
            await engine.start()
            try:
                results = await asyncio.gather(
                    engine.submit('a', 'hello', 1), engine.submit('b', 'world', 1)
                )
                self.assertEqual(results, ['hello', 'world'])
                self.assertEqual(engine.store.get('a')['original'], 'hello')
            finally:
                await engine.stop()
        self.assertEqual(engine._workers, [])

    async def test_bad_batch_fails_promptly(self):
        engine = self.engine(BrokenDetector(), batch_size=1)
        await engine.start()
        try:
            with (self.assertLogs('pii.engine', level='ERROR'), self.assertRaisesRegex(RuntimeError, 'inference failed')):
                await engine.submit('a', 'hello', 1)
        finally:
            await engine.stop()

    async def test_queue_full_returns_without_waiting(self):
        engine = self.engine()
        engine._queue = asyncio.Queue(maxsize=1)
        await engine.start()
        first = asyncio.create_task(engine.submit('a', 'hello', 1))
        # Keep the worker from consuming the queue for this saturation check.
        for worker in engine._workers:
            worker.cancel()
        await asyncio.gather(*engine._workers, return_exceptions=True)
        await asyncio.sleep(0)
        self.assertIsNone(await asyncio.wait_for(engine.submit('b', 'world', 1), 0.1))
        await engine.stop()
        with self.assertRaises(asyncio.CancelledError):
            await first

    async def test_stop_cancels_job_during_collection(self):
        engine = self.engine(batch_size=10, batch_timeout_s=10)
        await engine.start()
        task = asyncio.create_task(engine.submit('a', 'hello', 20))
        await asyncio.sleep(0.02)
        await engine.stop()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_stop_waits_for_inflight_inference(self):
        entered = threading.Event()
        release = threading.Event()

        class BlockingDetector:
            def predict_batch(self, texts):
                entered.set()
                release.wait(timeout=2)
                return [[] for _ in texts]

        engine = self.engine(BlockingDetector(), batch_size=1)
        await engine.start()
        task = asyncio.create_task(engine.submit('a', 'hello', 5))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            stopping = asyncio.create_task(engine.stop())
            await asyncio.sleep(0.02)
            self.assertFalse(stopping.done())
        finally:
            release.set()
            if 'stopping' in locals():
                await stopping
            else:
                await engine.stop()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_requires_start(self):
        with self.assertRaisesRegex(RuntimeError, 'not running'):
            await self.engine().submit('a', 'hello', 1)

    def test_zero_timeout_and_invalid_threads(self):
        self.assertEqual(self.engine(batch_timeout_s=0).batch_timeout_s, 0)
        with self.assertRaises(ValueError):
            InferenceEngine(detector=EmptyDetector(), threads=0)


if __name__ == '__main__':
    unittest.main()
