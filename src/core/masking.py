"""Reversible PII masking / unmasking.

Masking replaces every detected PII span with a fixed-width placeholder so the
overall character positions of the document are preserved (important for the
LLM to keep context and for the "positions preserved" criterion). The original
spans are stored alongside the masked text so unmasking restores the exact
original string.

The mask is deterministic for a given set of entities, so retries with the same
payload_id are idempotent.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ru_pii.schema import Entity

MASK_CHAR = "*"


@dataclass(frozen=True)
class MaskedSpan:
    start: int
    end: int
    text: str
    label: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _valid_span(start: int, end: int, length: int) -> bool:
    return type(start) is int and type(end) is int and 0 <= start < end <= length


def mask_text(text: str, entities: list[Entity]) -> tuple[str, list[MaskedSpan]]:
    """Return (masked_text, spans). Spans are sorted by start; offsets refer to
    the original text. Invalid model offsets fail instead of leaking unmasked text."""
    spans: list[MaskedSpan] = []
    for e in entities:
        if not _valid_span(e.start, e.end, len(text)):
            raise ValueError("invalid entity offsets")
        if e.label in {"PUBLIC_PERSON", "PUBLIC_ADDRESS"}:
            continue
        spans.append(MaskedSpan(e.start, e.end, text[e.start:e.end], e.label))
    spans.sort(key=lambda s: s.start)
    # Merge intervals so nested/overlapping predictions do not multiply work.
    intervals: list[tuple[int, int]] = []
    for span in spans:
        if intervals and span.start <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], span.end))
        else:
            intervals.append((span.start, span.end))
    chunks = []
    position = 0
    for start, end in intervals:
        chunks.extend((text[position:start], MASK_CHAR * (end - start)))
        position = end
    chunks.append(text[position:])
    return "".join(chunks), spans


def unmask_text(masked: str, spans: list[MaskedSpan]) -> str:
    """Restore the original text by splicing stored spans back into the mask.

    Offsets in spans refer to the original text; because masking preserved
    positions, the same offsets are valid in the masked string.
    """
    chars = list(masked)
    restored: dict[int, str] = {}
    for s in sorted(spans, key=lambda s: s.start):
        if not _valid_span(s.start, s.end, len(chars)) or len(s.text) != s.end - s.start:
            raise ValueError("invalid restoration span")
        for offset, char in enumerate(s.text, s.start):
            if offset in restored and restored[offset] != char:
                raise ValueError("conflicting restoration spans")
            restored[offset] = char
            chars[offset] = char
    return "".join(chars)


def spans_to_dicts(spans: list[MaskedSpan]) -> list[dict[str, Any]]:
    return [s.to_dict() for s in spans]


def spans_from_dicts(raw: list[dict[str, Any]]) -> list[MaskedSpan]:
    return [MaskedSpan(int(s["start"]), int(s["end"]), str(s["text"]), str(s["label"])) for s in raw]
