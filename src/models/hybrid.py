"""Hybrid detector: fast rules for pattern types + a model for semantic types.

Pattern-heavy PII (email, phone, card, CVV, PIN, INN, dates, passport, postal
code, IP, URL, SNILS, OMS) is caught by the near-zero-latency rule matcher. The
semantic types (PERSON, ADDRESS and its parts, citizenship, passport issuer,
cardholder, ...) are handled by the underlying model (GLiNER or the distilled
student). This keeps latency low while preserving quality on the hard types.
"""
from __future__ import annotations

from typing import Sequence

from ru_pii.schema import Entity

from .base import Detector
from .rules import rule_spans

# Labels handled by the rule matcher; the model is asked for the rest.
RULE_LABELS = frozenset(
    {
        "EMAIL", "URL", "IP_ADDRESS", "PHONE", "SNILS", "OMS", "CARD_NUMBER",
        "INN", "PASSPORT", "DRIVER_LICENSE", "BIRTH_DATE", "POSTAL_CODE",
        "CVV", "PIN",
    }
)


class HybridDetector:
    name = "hybrid"

    def __init__(self, model: Detector) -> None:
        self.model = model

    def predict(self, text: str) -> list[Entity]:
        return self._merge(text, self.model.predict(text))

    def predict_batch(self, texts: Sequence[str]) -> list[list[Entity]]:
        model_results = self.model.predict_batch(list(texts))
        return [self._merge(t, ents) for t, ents in zip(texts, model_results)]

    def _merge(self, text: str, model_entities: list[Entity]) -> list[Entity]:
        rule_ents = [Entity(s, e, label, 1.0, text[s:e]) for s, e, label in rule_spans(text)]
        # Keep model entities that are not pattern types (avoid double counting).
        kept = [e for e in model_entities if e.label not in RULE_LABELS]
        merged = rule_ents + kept
        merged.sort(key=lambda e: (e.start, e.end, e.label))
        return merged