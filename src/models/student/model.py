"""Distilled student model: a compact character-level BiLSTM-CRF.

The teacher is the trained GLiNER (deberta-v3-small) checkpoint. The student is
a tiny character-level sequence tagger that runs orders of magnitude faster on
CPU (no transformer, no tokenizer) while learning the same PII spans from the
corpus and from the teacher's soft labels.

Architecture:
    char embeddings -> BiLSTM -> linear -> CRF (BIO tagging over PII labels)

Inference is Viterbi decoding over characters; BIO tags are converted back to
character spans and then to :class:`ru_pii.schema.Entity`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ru_pii.schema import LABELS

# BIO tag scheme over the canonical labels.
_LABEL_IDS = {label: i for i, label in enumerate(sorted(LABELS))}
NUM_LABELS = len(_LABEL_IDS)
O_TAG = 0  # "O"
# tag id = 1 + 2*label_id (B) / 2 + 2*label_id (I)
NUM_TAGS = 1 + 2 * NUM_LABELS


def tag_id(label: str, prefix: str) -> int:
    return 1 + 2 * _LABEL_IDS[label] + (0 if prefix == "B" else 1)


def tag_to_label(tag: int) -> tuple[str, str] | None:
    if tag == O_TAG:
        return None
    idx = (tag - 1) // 2
    prefix = "B" if (tag - 1) % 2 == 0 else "I"
    return prefix, sorted(LABELS)[idx]


# Character vocabulary (covers Cyrillic, Latin, digits, punctuation, whitespace).
_CHARS = (
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    " +-.,:;!?()[]{}@#%&*_=/\\|'\"<>~^`"
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
)
# Deduplicate so ids stay within the embedding table.
_CHAR2ID = {c: i + 1 for i, c in enumerate(dict.fromkeys(_CHARS))}  # 0 = PAD/UNK
CHAR_VOCAB = len(_CHAR2ID) + 1


def char_id(ch: str) -> int:
    return _CHAR2ID.get(ch, 0)


@dataclass
class StudentConfig:
    char_vocab: int = CHAR_VOCAB
    num_tags: int = NUM_TAGS
    char_emb_dim: int = 32
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.3


class BiLSTMCRF(nn.Module):
    def __init__(self, cfg: StudentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.char_emb = nn.Embedding(cfg.char_vocab, cfg.char_emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            cfg.char_emb_dim,
            cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(cfg.hidden_dim * 2, cfg.num_tags)
        # CRF transition matrix [num_tags, num_tags]; trans[i, j] = score of i -> j.
        self.transitions = nn.Parameter(torch.randn(cfg.num_tags, cfg.num_tags) * 0.1)
        self._start = nn.Parameter(torch.randn(cfg.num_tags) * 0.1)
        self._end = nn.Parameter(torch.randn(cfg.num_tags) * 0.1)

    def _emissions(self, chars: torch.Tensor) -> torch.Tensor:
        emb = self.char_emb(chars)
        out, _ = self.lstm(emb)
        out = self.dropout(out)
        return self.fc(out)  # [B, T, num_tags]



    def decode(self, chars: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        """Viterbi decoding. Returns a list of tag sequences (one per batch item)."""
        emissions = self._emissions(chars)
        batch, seq, _ = emissions.shape
        out: list[list[int]] = []
        for b in range(batch):
            length = int(mask[b].sum().item())
            scores = emissions[b, :length]
            back: list[torch.Tensor] = []
            v = self._start + scores[0]
            for t in range(1, length):
                # v[j] = max_i (v[i] + trans[i,j]) + scores[t][j]
                trans = self.transitions  # [i, j]
                v_exp = v.unsqueeze(1) + trans  # [i, j]
                best, best_idx = v_exp.max(dim=0)
                back.append(best_idx)
                v = best + scores[t]
            v = v + self._end
            best_last = int(v.argmax().item())
            path = [best_last]
            for b_idx in reversed(back):
                path.append(int(b_idx[path[-1]].item()))
            path.reverse()
            out.append(path)
        return out


    @classmethod
    def load(cls, path: str | Path) -> "BiLSTMCRF":
        data = torch.load(Path(path), map_location="cpu")
        cfg = StudentConfig(**data["config"])
        model = cls(cfg)
        model.load_state_dict(data["state_dict"])
        model.eval()
        return model