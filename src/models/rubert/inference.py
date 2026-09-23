"""ONNX int8 inference for the fine-tuned rubert-tiny2 BIO tagger.

Reproduces the notebook inference (``train/notebooks/train_pii_masker.ipynb``):
chunked tokenization (max_len/stride), voting between overlapping chunks by
(start, end) token-offset key, BIO decoding, char-offset entities.

The exported ``model_int8.onnx`` takes ``input_ids`` + ``attention_mask`` and
returns tag logits ordered exactly like ``tags`` in ``model_config.json``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from ru_pii.schema import Entity

O_TAG = "O"


def bio_decode(offsets: Sequence[tuple[int, int]], tag_ids: Sequence[int], tags: Sequence[str]) -> list[tuple[int, int, str]]:
    """Decode per-token tag ids into char spans (start, end, label)."""
    spans: list[tuple[int, int, str]] = []
    cur: tuple[int, int, str] | None = None
    for (s, e), tid in zip(offsets, tag_ids):
        tag = tags[int(tid)]
        if tag == O_TAG:
            if cur is not None:
                spans.append(cur)
                cur = None
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "B" or cur is None or cur[2] != label:
            if cur is not None:
                spans.append(cur)
            cur = (s, e, label)
        else:
            cur = (cur[0], e, cur[2])
    if cur is not None:
        spans.append(cur)
    return spans


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class RubertOnnxInference:
    """Loads the ONNX int8 checkpoint directory and runs chunked BIO inference."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        batch_size: int = 16,
        max_len: int | None = None,
        stride: int | None = None,
    ) -> None:
        model_dir = Path(model_path)
        cfg = json.loads((model_dir / "model_config.json").read_text(encoding="utf-8"))
        self.tags: list[str] = list(cfg["tags"])
        self.public_labels: frozenset[str] = frozenset(cfg.get("public_labels", ()))
        self.max_len = int(max_len or cfg.get("max_len", 1024))
        self.stride = int(stride if stride is not None else cfg.get("stride", 128))
        self.batch_size = int(batch_size)
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        session_options = ort.SessionOptions()
        self._session = ort.InferenceSession(
            str(model_dir / "model_int8.onnx"),
            session_options,
            providers=["CPUExecutionProvider"],
        )
        input_names = [i.name for i in self._session.get_inputs()]
        if len(input_names) != 2:
            raise ValueError(f"unexpected ONNX inputs: {input_names!r}")
        self._input_ids_name, self._attention_mask_name = input_names

    def predict(self, text: str) -> list[Entity]:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: Sequence[str]) -> list[list[Entity]]:
        texts = list(texts)
        if not texts:
            return []
        enc = self._tokenizer(
            texts,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_len,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_attention_mask=True,
        )
        owner = enc["overflow_to_sample_mapping"]
        n_chunks = len(enc["input_ids"])
        votes: list[dict[tuple[int, int], tuple[int, float]]] = [dict() for _ in texts]
        order = sorted(range(n_chunks), key=lambda i: len(enc["input_ids"][i]))
        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        for i in range(0, len(order), self.batch_size):
            idxs = order[i : i + self.batch_size]
            width = max(len(enc["input_ids"][j]) for j in idxs)
            input_ids = np.full((len(idxs), width), pad_id, dtype=np.int64)
            attention_mask = np.zeros((len(idxs), width), dtype=np.int64)
            for bi, j in enumerate(idxs):
                ids = enc["input_ids"][j]
                input_ids[bi, : len(ids)] = ids
                attention_mask[bi, : len(ids)] = 1
            logits = self._session.run(
                None,
                {
                    self._input_ids_name: input_ids,
                    self._attention_mask_name: attention_mask,
                },
            )[0]
            probs = softmax(np.asarray(logits))
            conf = probs.max(axis=-1)
            pred = probs.argmax(axis=-1)
            for bi, j in enumerate(idxs):
                text_idx = int(owner[j])
                for k, (s, e) in enumerate(enc["offset_mapping"][j]):
                    if s == e:
                        continue
                    key = (s, e)
                    p = float(conf[bi, k])
                    cur = votes[text_idx].get(key)
                    if cur is None or p > cur[1]:
                        votes[text_idx][key] = (int(pred[bi, k]), p)
        results: list[list[Entity]] = []
        for text, vote in zip(texts, votes):
            score_by_start = {s: p for (s, _e), (_tid, p) in vote.items()}
            entities: list[Entity] = []
            for s, e, label in bio_decode(sorted(vote), [vote[k][0] for k in sorted(vote)], self.tags):
                if label in self.public_labels:
                    continue
                entities.append(Entity(s, e, label, score_by_start.get(s, 0.0), text[s:e], "rubert_onnx"))
            results.append(entities)
        return results
