"""Convert corpus records (char spans) to/from BIO tag sequences for the student.

Also provides the fast inference path: given raw text, run the BiLSTM-CRF and
return :class:`ru_pii.schema.Entity` objects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch

from ru_pii.schema import Entity

from .model import BiLSTMCRF, char_id, tag_id, tag_to_label, O_TAG


def text_to_chars(text: str, max_len: int = 512) -> list[int]:
    return [char_id(c) for c in text[:max_len]]


def spans_to_tags(text: str, spans: Iterable[tuple[int, int, str]], max_len: int = 512) -> list[int]:
    """Build BIO tags over characters from (start, end, label) spans."""
    n = min(len(text), max_len)
    tags = [O_TAG] * n
    for start, end, label in spans:
        start = max(0, start)
        end = min(n, end)
        if start >= end:
            continue
        tags[start] = tag_id(label, "B")
        for i in range(start + 1, end):
            tags[i] = tag_id(label, "I")
    return tags


def tags_to_entities(text: str, tags: list[int]) -> list[Entity]:
    """Convert a BIO tag sequence back to entities with char offsets."""
    entities: list[Entity] = []
    i = 0
    n = min(len(text), len(tags))
    while i < n:
        tag = tags[i]
        info = tag_to_label(tag)
        if info is None:
            i += 1
            continue
        prefix, label = info
        if prefix == "B":
            start = i
            j = i + 1
            while j < n and tag_to_label(tags[j]) == ("I", label):
                j += 1
            entities.append(Entity(start, j, label, 1.0, text[start:j]))
            i = j
        else:
            i += 1
    return entities


def collate(texts: list[str], max_len: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a batch of texts into (chars, mask) tensors."""
    ids = [text_to_chars(t, max_len) for t in texts]
    length = max(len(x) for x in ids)
    batch = len(ids)
    chars = torch.zeros(batch, length, dtype=torch.long)
    mask = torch.zeros(batch, length, dtype=torch.bool)
    for b, seq in enumerate(ids):
        chars[b, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[b, : len(seq)] = True
    return chars, mask


class StudentDetector:
    """Fast inference wrapper around the distilled BiLSTM-CRF."""

    def __init__(self, model: BiLSTMCRF, *, device: str = "cpu", max_len: int = 512) -> None:
        self.model = model
        self.device = device
        self.max_len = max_len
        self.model.to(device)
        self.model.eval()

    @classmethod
    def from_pretrained(cls, path: str | Path, *, device: str = "cpu", max_len: int = 512) -> "StudentDetector":
        return cls(BiLSTMCRF.load(path), device=device, max_len=max_len)

    def predict(self, text: str) -> list[Entity]:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[list[Entity]]:
        chars, mask = collate(texts, self.max_len)
        chars = chars.to(self.device)
        mask = mask.to(self.device)
        with torch.inference_mode():
            paths = self.model.decode(chars, mask)
        return [tags_to_entities(text, tags) for text, tags in zip(texts, paths)]