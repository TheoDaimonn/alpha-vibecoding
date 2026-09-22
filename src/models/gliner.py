"""GLiNER detector: wraps the trained RussianPIIDetector.

This is the high-quality teacher engine. It is slow on CPU (~16 texts/sec per
replica) but accurate. Used as the default until the distilled student is ready.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Sequence

from ru_pii.inference import RussianPIIDetector
from ru_pii.schema import Entity

MODEL_DIR = Path(__file__).resolve().parents[2] / "ru_gliner_hybrid_v4" / "models" / "gliner-ru-pii-small"


class GLiNERDetector:
    """A single loaded GLiNER instance with a thread-safe inference guard.

    GLiNER batch inference is not thread-safe on one tensor, so a replica
    serialises calls with a lock. Parallelism comes from running many replicas
    (one per worker process) and from batching many texts per call.
    """

    name = "gliner"

    def __init__(
        self,
        model_path: str | Path = MODEL_DIR,
        *,
        device: str = "cpu",
        batch_size: int = 16,
        threshold: float = 0.5,
        subtoken_budget: int = 512,
        word_window: int = 256,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.batch_size = batch_size
        self._lock = threading.Lock()
        self._detector = RussianPIIDetector.from_pretrained(
            self.model_path,
            device=device,
            threshold=threshold,
            batch_size=batch_size,
            subtoken_budget=subtoken_budget,
            word_window=word_window,
        )

    def predict(self, text: str) -> list[Entity]:
        with self._lock:
            return self._detector.predict(text)

    def predict_batch(self, texts: Sequence[str]) -> list[list[Entity]]:
        with self._lock:
            return self._detector.predict_batch(list(texts))