"""Construct detector strategies and apply optional post-processing once."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from ..core.config import Settings, settings
from .base import Detector
from .gliner import GLiNERDetector
from .hybrid import HybridDetector
from .postprocess import PostProcessedDetector
from .rubert_onnx_detector import RubertOnnxDetector
from .rules import RuleDetector
from .student_detector import StudentDetector
from .transformer_detector import TransformerDetector

DetectorBuilder = Callable[[Settings], Detector]


def _gliner(config: Settings) -> Detector:
    return GLiNERDetector(
        config.model_path,
        device=config.device,
        batch_size=config.model_batch_size,
        threshold=config.threshold,
    )


def _student(config: Settings) -> Detector:
    return StudentDetector(config.student_model_path, device=config.device)


def _transformer(config: Settings) -> Detector:
    return TransformerDetector(config.transformer_model_path, device=config.device)


def _rubert_onnx(config: Settings) -> Detector:
    return RubertOnnxDetector(config.rubert_model_path, batch_size=config.model_batch_size)


def _rules(config: Settings) -> Detector:
    return RuleDetector()


def _hybrid(config: Settings) -> Detector:
    return HybridDetector(_gliner(config))


_BUILDERS: Mapping[str, DetectorBuilder] = MappingProxyType({
    "gliner": _gliner,
    "student": _student,
    "transformer": _transformer,
    "rules": _rules,
    "rubert_onnx": _rubert_onnx,
    "hybrid": _hybrid,
})


def create_detector(
    detector: str | None = None,
    *,
    config: Settings | None = None,
    builders: Mapping[str, DetectorBuilder] | None = None,
) -> Detector:
    """Build a detector with injectable configuration and backend constructors.

    Existing callers use process settings. Tests and alternative deployments can
    supply their own settings/registry without mutating global state. A supplied
    registry replaces the defaults; hybrid uses a raw GLiNER backend by default.
    """
    config = settings if config is None else config
    registry = _BUILDERS if builders is None else builders
    kind = (config.detector if detector is None else detector).lower()
    try:
        builder = registry[kind]
    except KeyError:
        raise ValueError(f"unknown detector: {kind!r}") from None
    base = builder(config)
    return PostProcessedDetector(base) if config.postprocess else base
