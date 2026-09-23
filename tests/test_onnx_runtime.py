import os
import unittest

from src.core.masking import mask_text, unmask_text


@unittest.skipUnless(os.environ.get('RUN_ONNX_SMOKE') == '1', 'requires downloaded ONNX model')
class OnnxRuntimeTests(unittest.TestCase):
    def test_real_model_email_and_roundtrip(self):
        from src.models.rubert_onnx_detector import RubertOnnxDetector
        detector = RubertOnnxDetector('artifacts/rubert-tiny2-fine-tuning', batch_size=1,
                                     max_len=256, stride=64)
        email = 'test@example.com'
        texts = ['Клиент Иван Иванов, email: ' + email,
                 'Это обычный текст. ' * 100 + 'Контакт: ' + email, '']
        predictions = detector.predict_batch(texts)
        self.assertEqual(len(predictions), len(texts))
        for text, entities in zip(texts, predictions, strict=True):
            masked, spans = mask_text(text, entities)
            self.assertEqual(len(masked), len(text))
            self.assertEqual(unmask_text(masked, spans), text)
            if text:
                self.assertEqual(masked[-len(email):], '*' * len(email))

    def test_distil_model_email_and_roundtrip(self):
        from src.models.rubert_onnx_detector import RubertOnnxDetector
        detector = RubertOnnxDetector('artifacts/rubert-distil', batch_size=1,
                                      max_len=256, stride=64,
                                      onnx_filename='student_int8.onnx', name='distil')
        email = 'test@example.com'
        texts = ['Клиент Иван Иванов, email: ' + email,
                 'Это обычный текст. ' * 100 + 'Контакт: ' + email, '']
        predictions = detector.predict_batch(texts)
        self.assertEqual(len(predictions), len(texts))
        for text, entities in zip(texts, predictions, strict=True):
            masked, spans = mask_text(text, entities)
            self.assertEqual(len(masked), len(text))
            self.assertEqual(unmask_text(masked, spans), text)
            self.assertEqual(detector.name, 'distil')
