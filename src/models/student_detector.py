"""Distilled student detector: fast character-level BiLSTM-CRF.

Drop-in replacement for GLiNER. Much faster on CPU but slightly less accurate.
Enable with ``DETECTOR=student`` and point ``STUDENT_MODEL_PATH`` at the trained
checkpoint.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ru_pii.schema import Entity

from .student.inference import StudentDetector as _StudentDetector


class StudentDetector:
    name = "student"

    def __init__(self, model_path: str | Path, *, device: str = "cpu", max_len: int = 512) -> None:
        self._detector = _StudentDetector.from_pretrained(model_path, device=device, max_len=max_len)

    def predict(self, text: str) -> list[Entity]:
        return self._detector.predict(text)

    def predict_batch(self, texts: Sequence[str]) -> list[list[Entity]]:
        return self._detector.predict_batch(list(texts))