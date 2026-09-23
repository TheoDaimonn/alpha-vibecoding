"""Recall RuBERT detector served from an ONNX int8 checkpoint.

Enable with ``DETECTOR=recall_onnx`` and point ``RECALL_MODEL_PATH`` at the
exported model directory (``model_int8.onnx`` + ``encoder/`` + ``experiment.json``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ru_pii.schema import Entity

from .recall_onnx.inference import RecallOnnxInference


class RecallOnnxDetector:
    name = "recall_onnx"

    def __init__(
        self,
        model_path: str | Path,
        *,
        batch_size: int = 16,
        max_len: int = 1024,
        stride: int = 256,
        bias: float = 0.0,
        intra_threads: int = 1,
        inter_threads: int = 1,
    ) -> None:
        self._detector = RecallOnnxInference(
            model_path,
            batch_size=batch_size,
            max_len=max_len,
            stride=stride,
            bias=bias,
            intra_threads=intra_threads,
            inter_threads=inter_threads,
        )

    def predict(self, text: str) -> list[Entity]:
        return self._detector.predict(text)

    def predict_batch(self, texts: Sequence[str]) -> list[list[Entity]]:
        return self._detector.predict_batch(list(texts))