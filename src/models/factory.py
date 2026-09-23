"""Detector factory: build the configured engine.

Switching engines is a one-line config change (``DETECTOR`` env var):

    DETECTOR=gliner    -> GLiNER (default, high quality, slow on CPU)
    DETECTOR=student   -> distilled BiLSTM-CRF (fast, needs trained checkpoint)
    DETECTOR=rules     -> rule matcher only (fastest, pattern types only)
    DETECTOR=hybrid    -> rules for pattern types + model for semantic types
    DETECTOR=rubert_onnx -> fine-tuned rubert-tiny2 BIO tagger (ONNX int8)
"""
from __future__ import annotations

from .base import Detector
from ..core.config import settings
from .gliner import GLiNERDetector
from .hybrid import HybridDetector
from .postprocess import PostProcessedDetector
from .rules import RuleDetector
from .rubert_onnx_detector import RubertOnnxDetector
from .student_detector import StudentDetector
from .transformer_detector import TransformerDetector


def create_detector(detector: str | None = None) -> Detector:
    kind = (detector or settings.detector).lower()
    if kind == "gliner":
        base: Detector = GLiNERDetector(
            settings.model_path,
            device=settings.device,
            batch_size=settings.model_batch_size,
            threshold=settings.threshold,
        )
    elif kind == "student":
        base = StudentDetector(settings.student_model_path, device=settings.device)
    elif kind == "transformer":
        base = TransformerDetector(settings.transformer_model_path, device=settings.device)
    elif kind == "rubert_onnx":
        base = RubertOnnxDetector(
            settings.rubert_model_path,
            batch_size=settings.model_batch_size,
        )
    elif kind == "rules":
        base = RuleDetector()
    elif kind == "hybrid":
        model = create_detector(settings.detector if settings.detector != "hybrid" else "gliner")
        base = HybridDetector(model)
    else:
        raise ValueError(f"unknown detector: {kind!r}")
    if settings.postprocess:
        return PostProcessedDetector(base)
    return base