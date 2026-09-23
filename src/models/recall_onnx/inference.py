"""ONNX int8 inference for the recall RuBERT (per-type BIO heads).

The model is the one trained by experiments/pii_recall/train.py: a RuBERT-base
encoder with an independent O/B/I head per PII type. The exported ONNX graph
returns logits shaped (batch, seq, n_types, 3). Inference reproduces training:
windowed tokenization (max_len/stride), central-window logit selection per
absolute token, then per-type BIO decoding into char-offset entities.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from ru_pii.schema import Entity


class RecallOnnxInference:
    """Loads the recall ONNX int8 checkpoint and runs per-type BIO inference."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        batch_size: int = 16,
        max_len: int = 1024,
        stride: int = 256,
        bias: float = 0.0,
        intra_threads: int = 1,
        inter_threads: int = 1,
    ) -> None:
        model_dir = Path(model_path)
        cfg = json.loads((model_dir / "experiment.json").read_text(encoding="utf-8"))
        self.types: list[str] = list(cfg["labels"])
        self.max_len = int(max_len)
        self.stride = int(stride)
        self.batch_size = int(batch_size)
        self.bias = float(bias)
        if self.max_len < 1 or self.stride < 0 or self.stride >= self.max_len:
            raise ValueError("RECALL_MAX_LEN must be positive and RECALL_STRIDE in [0, max_len)")
        if self.batch_size < 1:
            raise ValueError("MODEL_BATCH_SIZE must be positive")
        if intra_threads < 1 or inter_threads < 1:
            raise ValueError("ONNX thread counts must be positive")
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir / "encoder", use_fast=True)
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = intra_threads
        session_options.inter_op_num_threads = inter_threads
        self._session = ort.InferenceSession(
            str(model_dir / "model_int8.onnx"),
            session_options,
            providers=["CPUExecutionProvider"],
        )
        input_names = [i.name for i in self._session.get_inputs()]
        if set(input_names) != {"input_ids", "attention_mask"}:
            raise ValueError(f"unexpected ONNX inputs: {input_names!r}")

    def predict(self, text: str) -> list[Entity]:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: Sequence[str]) -> list[list[Entity]]:
        texts = list(texts)
        if not texts:
            return []
        results: list[list[Entity]] = []
        for text in texts:
            windows = self._encode(text)
            tokens: dict[tuple[int, int], tuple[int, np.ndarray]] = {}
            for begin in range(0, len(windows), self.batch_size):
                batch = windows[begin : begin + self.batch_size]
                ids, mask = self._collate(batch)
                logits = self._session.run(None, {"input_ids": ids, "attention_mask": mask})[0]
                for window, scores in zip(batch, logits):
                    n = len(window["input_ids"])
                    for i, (s, e) in enumerate(window["offsets"]):
                        if s == e:
                            continue
                        centrality = min(i, n - 1 - i)
                        if (s, e) not in tokens or centrality > tokens[s, e][0]:
                            tokens[s, e] = (centrality, scores[i])
            offsets = sorted(tokens)
            if not offsets:
                results.append([])
                continue
            logits = np.stack([tokens[o][1] for o in offsets])
            results.append(self._decode(offsets, logits, text))
        return results

    def _encode(self, text: str) -> list[dict]:
        enc = self._tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        return [
            {"input_ids": ids, "offsets": offsets}
            for ids, offsets in zip(enc["input_ids"], enc["offset_mapping"])
        ]

    def _collate(self, windows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        width = max(len(w["input_ids"]) for w in windows)
        pad_id = self._tokenizer.pad_token_id or 0
        ids = np.full((len(windows), width), pad_id, dtype=np.int64)
        mask = np.zeros((len(windows), width), dtype=np.int64)
        for j, w in enumerate(windows):
            n = len(w["input_ids"])
            ids[j, :n] = w["input_ids"]
            mask[j, :n] = 1
        return ids, mask

    def _decode(self, offsets: list[tuple[int, int]], logits: np.ndarray, text: str) -> list[Entity]:
        scores = logits.copy()
        scores[:, :, 1:] += self.bias
        tags = scores.argmax(-1)
        entities: list[Entity] = []
        for c, label in enumerate(self.types):
            start = end = None
            for (s, e), tag in zip(offsets, tags[:, c]):
                if tag == 0 or tag == 1:
                    if start is not None:
                        entities.append(Entity(start, end, label, 1.0, text[start:end], "recall_onnx"))
                        start = end = None
                if tag:
                    if start is None:
                        start = s
                    end = e
            if start is not None:
                entities.append(Entity(start, end, label, 1.0, text[start:end], "recall_onnx"))
        return entities