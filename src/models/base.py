"""Detector abstraction.

Every detector (GLiNER, distilled student, rule-based, hybrid) implements this
protocol so the worker and the rest of the service never depend on a concrete
model. Swap the engine by changing the ``DETECTOR`` setting.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ru_pii.schema import Entity


@runtime_checkable
class Detector(Protocol):
    """A PII entity detector. Implementations must be thread-safe."""

    def predict(self, text: str) -> list[Entity]: ...

    def predict_batch(self, texts: Sequence[str]) -> list[list[Entity]]: ...

    @property
    def name(self) -> str: ...