import asyncio
import unittest
from dataclasses import replace
from unittest.mock import patch

from src.core import engine as engine_module
from src.core.correlation import MemoryCorrelationStore
from src.core.engine import InferenceEngine


class BoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_queue_character_limit_and_release(self):
        class Detector:
            def predict_batch(self, texts):
                return [[] for _ in texts]

        with patch.object(engine_module, 'settings', replace(engine_module.settings, worker_queue_chars=4)):
            engine = InferenceEngine(detector=Detector(), store=MemoryCorrelationStore(), threads=1, batch_size=1)
            await engine.start()
            try:
                self.assertIsNone(await engine.submit('a', '12345', 1))
                self.assertEqual(await engine.submit('b', '1234', 1), '1234')
                self.assertEqual(engine._pending_chars, 0)
            finally:
                await engine.stop()

    async def test_timed_out_queued_work_is_skipped(self):
        class Detector:
            def predict_batch(self, texts):
                raise AssertionError('expired job reached the model')

        engine = InferenceEngine(detector=Detector(), store=MemoryCorrelationStore(),
                                 threads=1, batch_size=2, batch_timeout_s=0.03)
        await engine.start()
        try:
            self.assertIsNone(await engine.submit('a', 'text', 0.001))
            await asyncio.sleep(0.05)
            self.assertIsNone(engine.store.get('a'))
            self.assertEqual(engine._pending_chars, 0)
        finally:
            await engine.stop()
