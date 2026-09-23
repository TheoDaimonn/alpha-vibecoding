"""Inference for the transformer student: tokenize -> BIO tags -> char spans."""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer

from ru_pii.schema import Entity

from ..batched import BatchedDetector

from .model import TransformerNER, tag_to_label, O_TAG


class TransformerDetector(BatchedDetector):
    """Fast inference wrapper around the rubert-tiny2 NER model."""

    def __init__(self, model: TransformerNER, *, device: str = "cpu", max_len: int = 256) -> None:
        self.model = model
        self.device = device
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model.cfg.model_name)
        self.model.to(device)
        self.model.eval()

    @classmethod
    def from_pretrained(cls, path: str | Path, *, device: str = "cpu", max_len: int = 256) -> "TransformerDetector":
        return cls(TransformerNER.load(path), device=device, max_len=max_len)

    def _predict_nonempty(self, texts: list[str]) -> list[list[Entity]]:
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_offsets_mapping=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        offsets = enc["offset_mapping"]  # list of lists of (start, end)

        with torch.inference_mode():
            logits = self.model(input_ids, attention_mask)
        preds = logits.argmax(dim=-1).cpu().tolist()

        out: list[list[Entity]] = []
        for text, tags, off in zip(texts, preds, offsets):
            out.append(self._tags_to_entities(text, tags, off))
        return out

    def _tags_to_entities(self, text: str, tags: list[int], offsets) -> list[Entity]:
        """Convert token-level BIO tags to char spans using offset mapping."""
        entities: list[Entity] = []
        i = 0
        n = len(tags)
        while i < n:
            start_char, end_char = int(offsets[i][0]), int(offsets[i][1])
            # Skip special tokens (offset (0,0)).
            if start_char == end_char:
                i += 1
                continue
            info = tag_to_label(tags[i])
            if info is None:
                i += 1
                continue
            prefix, label = info
            if prefix == "B":
                s = start_char
                j = i + 1
                while j < n:
                    js, je = int(offsets[j][0]), int(offsets[j][1])
                    if js == je:
                        j += 1
                        continue
                    jinfo = tag_to_label(tags[j])
                    if jinfo == ("I", label):
                        j += 1
                    else:
                        break
                e = int(offsets[j - 1][1]) if j - 1 >= 0 else end_char
                if s < e <= len(text):
                    entities.append(Entity(s, e, label, 1.0, text[s:e]))
                i = j
            else:
                i += 1
        return entities
