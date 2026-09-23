"""Detector wrapper that applies public-figure/address post-processing."""
from __future__ import annotations

from collections.abc import Sequence

from ru_pii.schema import Entity

from ..core.postprocess import filter_public
from .base import Detector


class PostProcessedDetector:
    """Wraps a base detector and filters out public figures / public addresses."""

    def __init__(self, base: Detector) -> None:
        self.base = base
        self._name = f"{getattr(base, 'name', 'base')}+post"

    @property
    def name(self) -> str:
        return self._name

    def validate_text(self, text: str) -> None:
        validate = getattr(self.base, "validate_text", None)
        if validate is not None:
            validate(text)

    def predict(self, text: str) -> list[Entity]:
        return filter_public(text, self.base.predict(text))

    def predict_batch(self, texts: Sequence[str]) -> list[list[Entity]]:
        results = self.base.predict_batch(list(texts))
        if len(results) != len(texts):
            raise ValueError("model returned an unexpected number of predictions")
        return [filter_public(t, ents) for t, ents in zip(texts, results, strict=True)]