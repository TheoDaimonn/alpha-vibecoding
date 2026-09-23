import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.config import Settings
from src.models.artifacts import require_model_file
from src.models.rules import _INN


class ValidationTests(unittest.TestCase):
    def test_invalid_settings(self):
        for kwargs in ({'worker_threads': 0}, {'worker_queue_size': 0}, {'result_ttl_s': 0},
                       {'request_timeout_s': float('nan')}, {'port': 70000},
                       {'correlation_store': 'redsi'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Settings(**kwargs)
        self.assertEqual(Settings(worker_batch_timeout_s=0).worker_batch_timeout_s, 0)

    def test_api_key_whitespace(self):
        with patch.dict('os.environ', {'API_KEYS': ' one , two, ,'}):
            self.assertEqual(Settings().api_keys, ('one', 'two'))

    def test_lfs_and_empty_model_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.onnx'
            for data in (b'', b'version https://git-lfs.github.com/spec/v1\noid sha256:abc'):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    require_model_file(path)
            path.write_bytes(b'binary-model')
            self.assertEqual(require_model_file(path), path)

    def test_inn_does_not_match_suffix_of_longer_number(self):
        self.assertIsNone(_INN.search('1234567890123'))
        self.assertIsNotNone(_INN.fullmatch('123456789012'))

    def test_student_covers_full_text_with_windows(self):
        from src.models.student.inference import StudentDetector
        from src.models.student.model import tag_id

        class Model:
            def to(self, device):
                return self

            def eval(self):
                return self

            def decode(self, chars, mask):
                return [[tag_id('PERSON', 'B')] + [tag_id('PERSON', 'I')] * (int(row.sum()) - 1)
                        for row in mask]

        detector = StudentDetector(Model())
        text = 'И' * 1500
        entities = detector.predict(text)
        self.assertEqual([(e.start, e.end) for e in entities], [(0, len(text))])
        self.assertEqual(entities[0].text, text)
        self.assertEqual(detector.predict(''), [])

    def test_multiple_memory_workers_rejected(self):
        from src import run_api
        with (patch.dict('os.environ', {'API_WORKERS': '2'}), patch.object(run_api, 'settings', Settings(correlation_store='memory')), self.assertRaisesRegex(ValueError, 'require')):
            run_api.main()

    def test_metrics_include_failures_in_total(self):
        from src.api.metrics import MetricsService
        metrics = MetricsService()
        metrics.record(masking=True, latency_s=0.5)
        metrics.record_error()
        metrics.record_overloaded()
        result = metrics.snapshot()
        self.assertEqual(result['total'], 3)
        self.assertEqual(result['successful'], 1)
        self.assertEqual(result['avg_latency_s'], 0.5)
