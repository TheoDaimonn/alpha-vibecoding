"""Tests for the rubert-tiny2 ONNX int8 detector.

Unit tests cover BIO decoding and the chunk-voting logic; integration tests
load the real exported checkpoint when it (and onnxruntime) are available.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.models.rubert.inference import bio_decode

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "rubert-tiny2-fine-tuning"
TAGS = ["O", "B-PERSON", "I-PERSON", "B-PHONE", "I-PHONE", "B-PUBLIC_PERSON", "I-PUBLIC_PERSON"]

try:
    import onnxruntime  # noqa: F401

    _HAS_ORT = True
except ImportError:
    _HAS_ORT = False

HAS_ARTIFACT = (ARTIFACT / "model_int8.onnx").is_file() and (ARTIFACT / "model_config.json").is_file()


class TestBioDecode:
    def test_simple_span(self):
        offsets = [(0, 3), (4, 7), (8, 11)]
        assert bio_decode(offsets, [1, 2, 0], TAGS) == [(0, 7, "PERSON")]

    def test_two_spans(self):
        offsets = [(0, 3), (4, 9), (10, 13)]
        assert bio_decode(offsets, [3, 4, 1], TAGS) == [(0, 9, "PHONE"), (10, 13, "PERSON")]

    def test_b_after_b_starts_new_span(self):
        offsets = [(0, 2), (2, 4)]
        assert bio_decode(offsets, [1, 1], TAGS) == [(0, 2, "PERSON"), (2, 4, "PERSON")]

    def test_i_without_b_opens_span(self):
        offsets = [(0, 2), (3, 5)]
        assert bio_decode(offsets, [2, 0], TAGS) == [(0, 2, "PERSON")]


@pytest.mark.skipif(not (_HAS_ORT and HAS_ARTIFACT), reason="onnxruntime or rubert artifact not available")
class TestRubertOnnxIntegration:
    @pytest.fixture(scope="class")
    def detector(self):
        from src.models.rubert_onnx_detector import RubertOnnxDetector

        return RubertOnnxDetector(ARTIFACT, batch_size=4)

    def test_detects_pii(self, detector):
        text = "Иванов Иван Иванович родился 12.05.1990, телефон +7 900 123-45-67, email test@example.com."
        entities = detector.predict(text)
        assert entities, "expected at least one detected entity"
        for e in entities:
            assert 0 <= e.start < e.end <= len(text)
            assert e.text == text[e.start : e.end]
            assert e.label not in {"PUBLIC_PERSON", "PUBLIC_ADDRESS"}
        labels = {e.label for e in entities}
        assert "PERSON" in labels or "PHONE" in labels or "EMAIL" in labels

    def test_public_person_not_returned(self, detector):
        text = "Стихи Александра Пушкина изучают в школе."
        entities = detector.predict(text)
        assert all(e.label != "PUBLIC_PERSON" for e in entities)
        assert all("Пушкин" not in e.text for e in entities)

    def test_predict_batch_matches_predict(self, detector):
        texts = [
            "Позвоните Иванову Ивану Ивановичу по номеру +7 495 123-45-67.",
            "Просто обычный текст без персональных данных.",
        ]
        batch = detector.predict_batch(texts)
        for text, entities in zip(texts, batch):
            single = detector.predict(text)
            # int8 batched inference may differ in confidence by a hair;
            # spans and labels must be identical.
            assert [(e.start, e.end, e.label) for e in entities] == [
                (e.start, e.end, e.label) for e in single
            ]

    def test_empty_and_long_inputs(self, detector):
        assert detector.predict("") == []
        long_text = ("Обычный текст. " * 2000) + "Иванов Иван Иванович"
        entities = detector.predict(long_text)
        assert entities, "expected detection in a long document"
        assert any(e.label == "PERSON" and e.end > 29000 for e in entities)
        for e in entities:
            assert 0 <= e.start < e.end <= len(long_text)
            assert e.text == long_text[e.start : e.end]


@pytest.mark.skipif(not (_HAS_ORT and HAS_ARTIFACT), reason="onnxruntime or rubert artifact not available")
def test_factory_creates_rubert_onnx():
    from src.models.factory import create_detector
    from src.models.rubert_onnx_detector import RubertOnnxDetector

    detector = create_detector("rubert_onnx")
    assert isinstance(detector, RubertOnnxDetector)
