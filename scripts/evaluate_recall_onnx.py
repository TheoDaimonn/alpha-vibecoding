"""Evaluate the ONNX int8 recall RuBERT on the test corpus.

Loads the quantized ONNX graph, runs the same windowed inference (max_length,
stride) with central-window logit selection as training, decodes per-type BIO
and reports exact typed-span metrics via ru_pii.span_metrics.

Usage:
    python scripts/evaluate_recall_onnx.py \
        --onnx artifacts/recall-rubert-1024/onnx/model_int8.onnx \
        --tokenizer artifacts/recall-rubert-1024/model/recall-model/encoder \
        --corpus ru_gliner_hybrid_v4/data/hybrid \
        --split test \
        --out artifacts/recall-rubert-1024/onnx/test_metrics_int8.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ru_gliner_hybrid_v4"))
from ru_pii.metrics import span_metrics  # noqa: E402
from ru_pii.schema import LABELS, read_jsonl  # noqa: E402

TYPES = sorted(LABELS)


def encode(text: str, tokenizer, max_length: int, stride: int) -> list[dict]:
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    windows = []
    for ids, offsets in zip(enc["input_ids"], enc["offset_mapping"]):
        windows.append({"input_ids": ids, "offsets": offsets})
    return windows


def collect_logits(sess, rows, tokenizer, max_length, stride, batch_size):
    """Central-window logits per absolute token; never truncate the document."""
    outputs = []
    for row in rows:
        windows = encode(row["text"], tokenizer, max_length, stride)
        tokens = {}
        for begin in range(0, len(windows), batch_size):
            batch = windows[begin : begin + batch_size]
            max_len = max(len(w["input_ids"]) for w in batch)
            ids = np.zeros((len(batch), max_len), dtype=np.int64)
            mask = np.zeros((len(batch), max_len), dtype=np.int64)
            for j, w in enumerate(batch):
                n = len(w["input_ids"])
                ids[j, :n] = w["input_ids"]
                mask[j, :n] = 1
            logits = sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]
            for window, scores in zip(batch, logits):
                n = len(window["input_ids"])
                for i, (s, e) in enumerate(window["offsets"]):
                    if s == e:
                        continue
                    centrality = min(i, n - 1 - i)
                    if (s, e) not in tokens or centrality > tokens[s, e][0]:
                        tokens[s, e] = (centrality, scores[i])
        offsets = sorted(tokens)
        outputs.append(
            (offsets, np.stack([tokens[o][1] for o in offsets]) if offsets else np.empty((0, len(TYPES), 3)))
        )
    return outputs


def decode(offsets, logits, bias=0.0):
    scores = logits.copy()
    scores[:, :, 1:] += bias
    tags = scores.argmax(-1)
    result = []
    for c, label in enumerate(TYPES):
        start = end = None
        for (s, e), tag in zip(offsets, tags[:, c]):
            if tag == 0 or tag == 1:
                if start is not None:
                    result.append({"start": start, "end": end, "label": label})
                    start = end = None
            if tag:
                if start is None:
                    start = s
                end = e
        if start is not None:
            result.append({"start": start, "end": end, "label": label})
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--bias", type=float, default=0.0)
    args = p.parse_args()

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    rows = read_jsonl(args.corpus / f"{args.split}.jsonl")

    started = time.time()
    outputs = collect_logits(sess, rows, tokenizer, args.max_length, args.stride, args.batch_size)
    elapsed = time.time() - started
    predictions = [decode(*x, args.bias) for x in outputs]
    metrics = span_metrics(rows, predictions)
    metrics["inference_elapsed_s"] = round(elapsed, 2)
    metrics["inference_docs_per_s"] = round(len(rows) / elapsed, 2)
    metrics["onnx"] = str(args.onnx)
    metrics["bias"] = args.bias

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics["exact_span_micro"], indent=2))
    print(f"docs/s: {metrics['inference_docs_per_s']}  elapsed: {metrics['inference_elapsed_s']}s")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()