"""Shared inference contract for models that operate on batches."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ru_pii.schema import Entity


class BatchedDetector(ABC):
    """Keep input order and bypass model inference for empty strings.

    Subclasses implement only the non-empty batch operation. Checking its
    cardinality prevents silently dropping requests when a backend misbehaves.
    """

    def predict(self, text: str) -> list[Entity]:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: Sequence[str]) -> list[list[Entity]]:
        results: list[list[Entity]] = [[] for _ in texts]
        positions = [i for i, text in enumerate(texts) if text]
        if not positions:
            return results
        predictions = self._predict_nonempty([texts[i] for i in positions])
        if len(predictions) != len(positions):
            raise RuntimeError("detector returned an unexpected number of predictions")
        for position, entities in zip(positions, predictions, strict=True):
            results[position] = entities
        return results

    @abstractmethod
    def _predict_nonempty(self, texts: list[str]) -> list[list[Entity]]:
        """Return one prediction per non-empty input, in the same order."""
