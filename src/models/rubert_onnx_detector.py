"""Fine-tuned rubert-tiny2 detector served from an ONNX int8 checkpoint.

Enable with ``DETECTOR=rubert_onnx`` and point ``RUBERT_MODEL_PATH`` at the
exported model directory (``model_int8.onnx`` + tokenizer + ``model_config.json``).
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ru_pii.schema import Entity

from .rubert.inference import RubertOnnxInference
from .rules import supplement_emails


class RubertOnnxDetector:
    name = "rubert_onnx"

    def __init__(
        self,
        model_path: str | Path,
        *,
        batch_size: int = 16,
        max_len: int | None = None,
        stride: int | None = None,
        intra_threads: int = 1,
        inter_threads: int = 1,
        onnx_filename: str = "model_int8.onnx",
        name: str = "rubert_onnx",
    ) -> None:
        self.name = name
        self._detector = RubertOnnxInference(
            model_path,
            batch_size=batch_size,
            max_len=max_len,
            stride=stride,
            intra_threads=intra_threads,
            inter_threads=inter_threads,
            onnx_filename=onnx_filename,
            source=name,
        )

    def predict(self, text: str) -> list[Entity]:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: Sequence[str]) -> list[list[Entity]]:
        texts = list(texts)
        predictions = self._detector.predict_batch(texts)
        return [supplement_emails(text, entities) for text, entities in zip(texts, predictions, strict=True)]
