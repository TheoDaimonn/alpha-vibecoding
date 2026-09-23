"""Transformer detector: drop-in replacement for GLiNER/student.

Enable with ``DETECTOR=transformer`` and point ``TRANSFORMER_MODEL_PATH`` at the
trained rubert-tiny2 checkpoint.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ru_pii.schema import Entity

from .transformer.inference import TransformerDetector as _TransformerDetector


class TransformerDetector:
    name = "transformer"

    def __init__(self, model_path: str | Path, *, device: str = "cpu", max_len: int = 256) -> None:
        self._detector = _TransformerDetector.from_pretrained(model_path, device=device, max_len=max_len)

    def predict(self, text: str) -> list[Entity]:
        return self._detector.predict(text)

    def predict_batch(self, texts: Sequence[str]) -> list[list[Entity]]:
        return self._detector.predict_batch(list(texts))