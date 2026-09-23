"""Detector composition and batch contracts, without network or checkpoints."""
from dataclasses import replace

import pytest
import torch

from ru_pii.schema import Entity
from src.core.config import Settings
from src.models import factory
from src.models.batched import BatchedDetector
from src.models.hybrid import HybridDetector
from src.models.postprocess import PostProcessedDetector
from src.models.student.inference import StudentDetector
from src.models.student.model import BiLSTMCRF, StudentConfig


class EchoDetector(BatchedDetector):
    name = "echo"

    def _predict_nonempty(self, texts):
        return [[Entity(0, len(text), "PERSON", 1.0, text)] for text in texts]


def test_batch_preserves_positions_and_single_prediction():
    detector = EchoDetector()
    texts = ("", "Иван", "", "Анна", "")
    results = detector.predict_batch(texts)
    assert [bool(result) for result in results] == [False, True, False, True, False]
    assert results[1] == detector.predict("Иван")
    assert results[3][0].text == "Анна"
    assert results[0] is not results[2]


@pytest.mark.parametrize("texts", [[], [""], ["", ""]])
def test_empty_inputs_do_not_call_backend(texts):
    class NoBackend(BatchedDetector):
        def _predict_nonempty(self, texts):
            pytest.fail("empty inputs must bypass inference")

    assert NoBackend().predict_batch(texts) == [[] for _ in texts]


def test_incomplete_backend_batch_fails_explicitly():
    class Broken(BatchedDetector):
        def _predict_nonempty(self, texts):
            return []

    with pytest.raises(RuntimeError, match="number of predictions"):
        Broken().predict_batch(["Иван"])


def test_student_mixed_empty_batch_matches_nonempty_batch():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(42)
            detector = StudentDetector(BiLSTMCRF(StudentConfig(char_emb_dim=4, hidden_dim=4)))
            expected = detector.predict_batch(["Иван", "Анна"])
            assert detector.predict_batch(["", "Иван", "", "Анна"]) == [[], expected[0], [], expected[1]]
            assert detector.predict_batch([]) == []
    finally:
        torch.set_num_threads(old_threads)


def test_factory_uses_injected_settings_and_registry():
    config = Settings(detector="custom", postprocess=False)
    seen = []
    detector = EchoDetector()

    def build(received):
        seen.append(received)
        return detector

    assert factory.create_detector(config=config, builders={"custom": build}) is detector
    assert seen == [config]
    wrapped = factory.create_detector(
        "CUSTOM", config=replace(config, postprocess=True), builders={"custom": build}
    )
    assert isinstance(wrapped, PostProcessedDetector)
    assert wrapped.base is detector


def test_factory_rejects_unknown_before_constructing():
    with pytest.raises(ValueError, match="unknown detector"):
        factory.create_detector("missing", builders={})


def test_hybrid_applies_postprocessing_once(monkeypatch):
    detector = EchoDetector()
    monkeypatch.setattr(factory, "_gliner", lambda config: detector)
    result = factory.create_detector(config=Settings(detector="hybrid", postprocess=True))
    assert isinstance(result, PostProcessedDetector)
    assert isinstance(result.base, HybridDetector)
    assert result.base.model is detector
