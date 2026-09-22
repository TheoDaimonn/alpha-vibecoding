"""Detector factory: build the configured engine.

Switching engines is a one-line config change (``DETECTOR`` env var):

    DETECTOR=gliner    -> GLiNER (default, high quality, slow on CPU)
    DETECTOR=student   -> distilled BiLSTM-CRF (fast, needs trained checkpoint)
    DETECTOR=rules     -> rule matcher only (fastest, pattern types only)
    DETECTOR=hybrid    -> rules for pattern types + model for semantic types
"""
from __future__ import annotations

from .base import Detector
from ..core.config import settings
from .gliner import GLiNERDetector
from .hybrid import HybridDetector
from .rules import RuleDetector
from .student_detector import StudentDetector


def create_detector(detector: str | None = None) -> Detector:
    kind = (detector or settings.detector).lower()
    if kind == "gliner":
        return GLiNERDetector(
            settings.model_path,
            device=settings.device,
            batch_size=settings.model_batch_size,
            threshold=settings.threshold,
        )
    if kind == "student":
        return StudentDetector(settings.student_model_path, device=settings.device)
    if kind == "rules":
        return RuleDetector()
    if kind == "hybrid":
        model = create_detector(settings.detector if settings.detector != "hybrid" else "gliner")
        return HybridDetector(model)
    raise ValueError(f"unknown detector: {kind!r}")