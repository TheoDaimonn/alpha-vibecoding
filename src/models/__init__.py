"""Model layer: pluggable PII detectors.

The service talks to a :class:`Detector` protocol. Engines are swappable via the
``DETECTOR`` setting (see :func:`create_detector`):

    gliner   -> high-quality teacher (default)
    student  -> distilled fast BiLSTM-CRF
    rules    -> rule matcher only (pattern types)
    hybrid   -> rules + model
"""
from __future__ import annotations

from .base import Detector
from .factory import create_detector
from .gliner import GLiNERDetector, MODEL_DIR
from .hybrid import HybridDetector
from .rules import RuleDetector, rule_spans
from .student_detector import StudentDetector

__all__ = [
    "Detector",
    "create_detector",
    "GLiNERDetector",
    "HybridDetector",
    "RuleDetector",
    "StudentDetector",
    "rule_spans",
    "MODEL_DIR",
]