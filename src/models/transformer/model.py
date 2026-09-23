"""Transformer student: rubert-tiny2 with a token classification head.

The teacher is GLiNER (deberta-v3-small). The student is a small Russian
transformer (rubert-tiny2, 29M params) fine-tuned for token-level BIO tagging.
It runs ~1800 texts/sec on CPU (batch 32) — fast enough for 1k RPS — while
learning contextual PII detection (distinguishing "поэт Пушкин" from a client).

Architecture:
    rubert-tiny2 (frozen or fine-tuned) -> linear head -> BIO tags

Inference: tokenize -> model -> BIO tags -> char spans (via offset mapping).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from ru_pii.schema import LABELS

# BIO tag scheme (same as the char-level student).
_LABEL_IDS = {label: i for i, label in enumerate(sorted(LABELS))}
NUM_LABELS = len(_LABEL_IDS)
O_TAG = 0
NUM_TAGS = 1 + 2 * NUM_LABELS


def tag_id(label: str, prefix: str) -> int:
    return 1 + 2 * _LABEL_IDS[label] + (0 if prefix == "B" else 1)


def tag_to_label(tag: int) -> tuple[str, str] | None:
    if tag == O_TAG:
        return None
    idx = (tag - 1) // 2
    prefix = "B" if (tag - 1) % 2 == 0 else "I"
    return prefix, sorted(LABELS)[idx]


@dataclass
class TransformerConfig:
    model_name: str = "cointegrated/rubert-tiny2"
    num_tags: int = NUM_TAGS
    dropout: float = 0.1
    max_len: int = 256


class TransformerNER(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = AutoModel.from_pretrained(cfg.model_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(hidden, cfg.num_tags)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.classifier(self.dropout(out))  # [B, T, num_tags]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "config": self.cfg.__dict__}, path)

    @classmethod
    def load(cls, path: str | Path) -> "TransformerNER":
        data = torch.load(Path(path), map_location="cpu")
        cfg = TransformerConfig(**data["config"])
        model = cls(cfg)
        model.load_state_dict(data["state_dict"])
        model.eval()
        return model