import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException, Response

from src.api import main
from src.api.schemas import ProcessRequest
from src.core.correlation import MemoryCorrelationStore


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_overload_preserves_retry_header(self):
        engine = SimpleNamespace(store=MemoryCorrelationStore(), submit=AsyncMock(return_value=None))
        with (patch.object(main, '_engine', engine), self.assertRaises(HTTPException) as raised):
            await main.process(ProcessRequest(payload_id='test', payload='hello'), None, Response())
        self.assertEqual(raised.exception.status_code, 429)
        self.assertEqual(raised.exception.headers, {'Retry-After': '1'})

    async def test_lifespan_stops_engine_after_exception(self):
        engine = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
        with (patch.object(main, '_engine', engine), self.assertRaisesRegex(RuntimeError, 'test failure')):
            async with main.lifespan(main.app):
                raise RuntimeError('test failure')
        engine.start.assert_awaited_once()
        engine.stop.assert_awaited_once()
