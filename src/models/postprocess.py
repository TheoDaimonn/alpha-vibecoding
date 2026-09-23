"""Detector wrapper that applies public-figure/address post-processing."""
from __future__ import annotations

from typing import Sequence

from ru_pii.schema import Entity

from ..core.postprocess import filter_public
from .base import Detector


class PostProcessedDetector:
    """Wraps a base detector and filters out public figures / public addresses."""

    def __init__(self, base: Detector) -> None:
        self.base = base
        self.name = getattr(base, "name", "base") + "+post"

    def predict(self, text: str) -> list[Entity]:
        return filter_public(text, self.base.predict(text))

    def predict_batch(self, texts: Sequence[str]) -> list[list[Entity]]:
        return [filter_public(t, ents) for t, ents in zip(texts, self.base.predict_batch(list(texts)))]