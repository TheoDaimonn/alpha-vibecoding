"""Construct detector strategies and apply optional post-processing once."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from ..core.config import Settings, settings
from .base import Detector
from .hybrid import HybridDetector
from .postprocess import PostProcessedDetector
from .rules import RuleDetector

DetectorBuilder = Callable[[Settings], Detector]


def _student(config: Settings) -> Detector:
    from .student_detector import StudentDetector

    return StudentDetector(config.student_model_path, device=config.device)


def _rubert_onnx(config: Settings) -> Detector:
    from .rubert_onnx_detector import RubertOnnxDetector

    return RubertOnnxDetector(
        config.rubert_model_path, batch_size=config.model_batch_size,
        max_len=config.rubert_max_len, stride=config.rubert_stride,
        intra_threads=config.onnx_intra_threads, inter_threads=config.onnx_inter_threads,
    )


def _distil(config: Settings) -> Detector:
    from .rubert_onnx_detector import RubertOnnxDetector

    return RubertOnnxDetector(
        config.distil_model_path, batch_size=config.model_batch_size,
        max_len=config.rubert_max_len, stride=config.rubert_stride,
        intra_threads=config.onnx_intra_threads, inter_threads=config.onnx_inter_threads,
        onnx_filename="student_int8.onnx", name="distil",
    )


def _rules(config: Settings) -> Detector:
    return RuleDetector()


def _hybrid(config: Settings) -> Detector:
    return HybridDetector(_student(config))


_BUILDERS: Mapping[str, DetectorBuilder] = MappingProxyType({
    "student": _student,
    "rules": _rules,
    "rubert_onnx": _rubert_onnx,
    "distil": _distil,
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
    registry replaces the defaults; hybrid uses a raw student backend by default.
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
