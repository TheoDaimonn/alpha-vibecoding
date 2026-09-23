"""Rule-based fast matcher for pattern-heavy PII types.

Many PII types are highly regular (email, phone, card number, CVV, PIN, INN,
dates, passport series/number, postal code, IP, URL, SNILS, OMS). Detecting
them with regex is near-zero latency and does not need a neural network. This
matcher is combined with a model in the hybrid detector.

All matchers are case-insensitive and return (start, end, label) spans.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from ru_pii.schema import Entity

# --- compiled patterns ---
_EMAIL = re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9._%+-])")
_PHONE = re.compile(
    r"(?<!\d)(?:\+7|8|7)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\d)"
)
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_CVV = re.compile(r"(?<!\d)\d{3}(?!\d)")
_PIN = re.compile(r"(?<!\d)\d{4}(?!\d)")
_INN = re.compile(r"(?<!\d)(?:\d{10}|\d{12})(?!\d)")
_POSTAL = re.compile(r"(?<!\d)\d{6}(?!\d)")
_IP = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
_URL = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"']+")
_SNILS = re.compile(r"(?<!\d)\d{3}[- ]?\d{3}[- ]?\d{3}[- ]?\d{2}(?!\d)")
_OMS = re.compile(r"(?<!\d)\d{16}(?!\d)")
_PASSPORT = re.compile(r"(?<!\d)\d{2}[ ]?\d{2}[ ]?\d{6}(?!\d)")
_PASSPORT_SERIES = re.compile(r"(?<!\d)\d{2}[ ]?\d{2}(?!\d)")
_PASSPORT_NUMBER = re.compile(r"(?<!\d)\d{6}(?!\d)")
_DRIVER_LICENSE = re.compile(r"(?<!\d)\d{2}[ ]?\d{2}[ ]?\d{6}(?!\d)")
_DATE = re.compile(
    r"(?<!\d)(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}[./-]\d{1,2}[./-]\d{1,2})(?!\d)"
)

# Ordered list of (pattern, label). First match wins; longer/more specific first.
_RULES: list[tuple[re.Pattern[str], str]] = [
    (_EMAIL, "EMAIL"),
    (_URL, "URL"),
    (_IP, "IP_ADDRESS"),
    (_PHONE, "PHONE"),
    (_SNILS, "SNILS"),
    (_OMS, "OMS"),
    (_CARD, "CARD_NUMBER"),
    (_INN, "INN"),
    (_PASSPORT, "PASSPORT"),
    (_DRIVER_LICENSE, "DRIVER_LICENSE"),
    (_DATE, "BIRTH_DATE"),
    (_POSTAL, "POSTAL_CODE"),
    (_CVV, "CVV"),
    (_PIN, "PIN"),
]


def _dedupe(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Drop any span fully contained in a higher-priority span.

    This removes e.g. CVV/PIN inside a card number and PASSPORT inside an INN.
    """
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    out: list[tuple[int, int, str]] = []
    for s in spans:
        if out and s[0] >= out[-1][0] and s[1] <= out[-1][1]:
            continue
        out.append(s)
    return out


def rule_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for pattern, label in _RULES:
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end(), label))
    return _dedupe(spans)


def supplement_emails(text: str, entities: list[Entity]) -> list[Entity]:
    """Cover complete email spans when a BIO model predicts only a fragment."""
    result = list(entities)
    for match in _EMAIL.finditer(text):
        start, end = match.span()
        if not any(entity.start <= start and entity.end >= end for entity in entities):
            result.append(Entity(start, end, "EMAIL", 1.0, match.group(), "rules"))
    return sorted(result, key=lambda entity: (entity.start, entity.end, entity.label))


class RuleDetector:
    """Detector that only uses the fast rule matcher (no model)."""

    name = "rules"

    def predict(self, text: str) -> list[Entity]:
        return [Entity(s, e, label, 1.0, text[s:e]) for s, e, label in rule_spans(text)]

    def predict_batch(self, texts: Iterable[str]) -> list[list[Entity]]:
        return [self.predict(t) for t in texts]