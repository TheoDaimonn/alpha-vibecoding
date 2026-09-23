import asyncio
import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from fastapi import HTTPException, Response
from starlette.requests import Request

from ru_pii.schema import Entity
from src.api import main
from src.api.schemas import ProcessRequest, ProcessResponse
from src.core.correlation import MemoryCorrelationStore
from src.core.engine import InferenceEngine


class NameDetector:
    def predict_batch(self, texts):
        results = []
        for text in texts:
            start = text.find('Иван')
            results.append([Entity(start, start + 4, 'PERSON', 1, 'Иван')] if start >= 0 else [])
        return results


def request(key=''):
    return Request({'type': 'http', 'headers': [(b'x-api-key', key.encode())]})


class ProcessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = InferenceEngine(detector=NameDetector(), store=MemoryCorrelationStore(),
                                      threads=2, batch_size=8, batch_timeout_s=0.001)
        self.engine_patch = patch.object(main, '_engine', self.engine)
        self.settings_patch = patch.object(main, 'settings', replace(main.settings, api_keys=()))
        self.engine_patch.start()
        self.settings_patch.start()
        await self.engine.start()

    async def asyncTearDown(self):
        await self.engine.stop()
        self.engine_patch.stop()
        self.settings_patch.stop()

    async def process(self, text, pid='id', key=''):
        result = await main.process(ProcessRequest(payload=text, payload_id=pid), request(key), Response())
        return result.result

    async def test_contract_has_only_original_fields(self):
        self.assertEqual(set(ProcessRequest.model_fields), {'payload', 'payload_id'})
        self.assertEqual(set(ProcessResponse.model_fields), {'result'})
        self.assertTrue(all(field.is_required() for field in ProcessRequest.model_fields.values()))

    async def test_exact_roundtrip_and_retries(self):
        original = '🙂 Иван\n* конец'
        masked = await self.process(original)
        self.assertEqual(masked, '🙂 ****\n* конец')
        for _ in range(3):
            self.assertEqual(await self.process(original), masked)
            self.assertEqual(await self.process(masked), original)

    async def test_empty_and_no_entities(self):
        for i, text in enumerate(('', 'hello', '***', '🙂\n')):
            self.assertEqual(await self.process(text, str(i)), text)
            self.assertEqual(await self.process(text, str(i)), text)

    async def test_unmask_requires_saved_mask(self):
        await self.process('Иван дома')
        for text in ('**** тут', 'произвольная строка', ''):
            self.assertEqual(await self.process(text), '**** дома')
        self.assertEqual(self.engine.store.get('id')['original'], 'Иван дома')

    async def test_existing_id_does_not_invoke_model(self):
        await self.process('Иван дома')
        with patch.object(self.engine, 'submit', side_effect=AssertionError('inference on unmask')):
            self.assertEqual(await self.process('**** дома'), 'Иван дома')

    async def test_concurrent_same_original(self):
        results = await asyncio.gather(*(self.process('Иван дома') for _ in range(25)))
        self.assertEqual(set(results), {'**** дома'})
        self.assertEqual(await self.process('**** дома'), 'Иван дома')

    async def test_concurrent_writes_preserve_first_original(self):
        texts = ['Иван дома', 'Иван тут']
        results = await asyncio.gather(*(self.process(text) for text in texts))
        record = self.engine.store.get('id')
        for text, result in zip(texts, results, strict=True):
            self.assertEqual(result, record['original'] if text == record['masked'] else record['masked'])
        self.assertEqual(await self.process(record['masked']), record['original'])

    async def test_api_key_does_not_change_correlation_id(self):
        with patch.object(main, 'settings', replace(main.settings, api_keys=('one', 'two'))):
            self.assertEqual(await self.process('Иван дома', key='one'), '**** дома')
            self.assertEqual(await self.process('**** дома', key='two'), 'Иван дома')
        self.assertIn('id', self.engine.store._data)

    async def test_http_roundtrip_authentication_and_validation(self):
        from asgi_client import call_asgi
        with patch.object(main, 'settings', replace(main.settings, api_keys=('secret',))):
            body = json.dumps({'payload': 'Иван', 'payload_id': 'http'}).encode()
            code, _, _ = await call_asgi(main.app, body)
            self.assertEqual(code, 401)
            headers = [(b'x-api-key', b'secret')]
            code, _, data = await call_asgi(main.app, body, headers=headers)
            self.assertEqual(code, 200)
            self.assertEqual(set(json.loads(data)), {'result'})
            body = json.dumps({'payload': '****', 'payload_id': 'http'}).encode()
            code, _, data = await call_asgi(main.app, body, headers=headers)
            self.assertEqual(code, 200)
            self.assertEqual(json.loads(data)['result'], 'Иван')
            code, _, _ = await call_asgi(main.app, b'{"payload": "x"}', headers=headers)
            self.assertEqual(code, 422)

    async def test_health_reports_storage_failure(self):
        self.assertEqual(await main.health(), {'status': 'ok'})
        with (patch.object(self.engine.store, 'ping', side_effect=ConnectionError('private credentials')),
              self.assertRaises(HTTPException) as caught):
            await main.health()
        self.assertEqual(caught.exception.status_code, 503)
        self.assertNotIn('private', caught.exception.detail)
