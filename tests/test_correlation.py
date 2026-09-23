import os
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from src.core.correlation import (
    MemoryCorrelationStore,
    RedisCorrelationStore,
    create_correlation_store,
)


def record(text):
    return {'original': text, 'masked': '*' * len(text), 'spans': []}


class MemoryStoreTests(unittest.TestCase):
    def test_ttl_and_cleanup(self):
        with patch('src.core.correlation.monotonic') as clock:
            clock.return_value = 10
            store = MemoryCorrelationStore(ttl_s=2)
            store.put_if_absent('a', record('one'))
            clock.return_value = 11
            store.put_if_absent('b', record('two'))
            self.assertIsNotNone(store.get('a'))
            clock.return_value = 12
            self.assertIsNone(store.get('a'))
            self.assertNotIn('a', store._data)
            self.assertIsNotNone(store.get('b'))
            clock.return_value = 13
            self.assertIsNone(store.get_result('b'))

    def test_first_writer_wins_concurrently(self):
        store = MemoryCorrelationStore()
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda n: store.put_if_absent('same', record(str(n))), range(100)))
        self.assertTrue(all(result == results[0] for result in results))
        self.assertEqual(store.get('same'), results[0])

    def test_returned_values_cannot_mutate_store(self):
        store = MemoryCorrelationStore()
        data = record('one')
        result = store.put_if_absent('a', data)
        data['original'] = 'changed'
        result['spans'].append('changed')
        self.assertEqual(store.get('a'), record('one'))

    def test_unknown_backend_and_zero_ttl_rejected(self):
        with self.assertRaises(ValueError):
            create_correlation_store('redsi')
        with self.assertRaises(ValueError):
            MemoryCorrelationStore(ttl_s=0)


@unittest.skipUnless(os.environ.get('REDIS_TEST_URL'), 'requires isolated Redis')
class RedisStoreTests(unittest.TestCase):
    def test_atomic_write_and_expiration(self):
        store = RedisCorrelationStore(url=os.environ['REDIS_TEST_URL'], ttl_s=60)
        key = 'audit-' + uuid.uuid4().hex
        redis_key = 'corr:' + key
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda n: store.put_if_absent(key, record(str(n))), range(100)))
            self.assertTrue(all(result == results[0] for result in results))
            self.assertEqual(store.get(key), results[0])
            self.assertGreater(store._store._client.ttl(redis_key), 0)
            store._store._client.pexpire(redis_key, 1)
            import time
            time.sleep(0.02)
            self.assertIsNone(store.get(key))
        finally:
            store._store._client.delete(redis_key)


@unittest.skipUnless(os.environ.get('REDIS_TEST_URL'), 'requires isolated Redis')
class AsyncRedisTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_reads_and_connection_close(self):
        from src.api.correlation_service import CorrelationService
        store = RedisCorrelationStore(url=os.environ['REDIS_TEST_URL'], ttl_s=60)
        key = 'audit-' + uuid.uuid4().hex
        try:
            store.put_if_absent(key, record('test'))
            service = CorrelationService(store)
            self.assertTrue(await service.ping())
            self.assertEqual(await service.get(key), record('test'))
        finally:
            store._store._client.delete('corr:' + key)
            await store.aclose()
