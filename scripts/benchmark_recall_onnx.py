"""Benchmark ONNX int8 recall RuBERT throughput on CPU.

Measures full-window inference (tokenization + ONNX + BIO decode) in texts/sec
for several batch sizes, mirroring how the service batches requests. Reports
p50/p95 latency and achieved texts-per-second.

Usage:
    python scripts/benchmark_recall_onnx.py \
        --onnx artifacts/recall-rubert-1024/onnx/model_int8.onnx \
        --tokenizer artifacts/recall-rubert-1024/model/recall-model/encoder \
        --out experiments/pii_recall/benchmark_recall_onnx.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ru_gliner_hybrid_v4"))
from ru_pii.schema import LABELS  # noqa: E402

TYPES = sorted(LABELS)

TEXTS = [
    f"Клиент Иванов Иван Иванович, email: user{i}@example.org, паспорт 4509 {i:06d}, "
    f"телефон +7 900 123-45-67, ИНН 7707083893, адрес: г. Москва, ул. Тверская, д. {i}"
    for i in range(64)
]


def encode(text: str, tokenizer, max_length: int, stride: int) -> list[dict]:
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    return [{"input_ids": ids, "offsets": offsets} for ids, offsets in zip(enc["input_ids"], enc["offset_mapping"])]


def predict_batch(sess, tokenizer, texts, max_length, stride):
    """Run windowed inference over a batch of texts and return decoded spans."""
    results = []
    for text in texts:
        windows = encode(text, tokenizer, max_length, stride)
        tokens = {}
        for w in windows:
            n = len(w["input_ids"])
            ids = np.asarray([w["input_ids"]], dtype=np.int64)
            mask = np.ones_like(ids)
            logits = sess.run(None, {"input_ids": ids, "attention_mask": mask})[0][0]
            for i, (s, e) in enumerate(w["offsets"]):
                if s == e:
                    continue
                centrality = min(i, n - 1 - i)
                if (s, e) not in tokens or centrality > tokens[s, e][0]:
                    tokens[s, e] = (centrality, logits[i])
        offsets = sorted(tokens)
        scores = np.stack([tokens[o][1] for o in offsets]) if offsets else np.empty((0, len(TYPES), 3))
        results.append(decode(offsets, scores))
    return results


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


def measure(sess, tokenizer, batch, max_length, stride, runs=30):
    samples = []
    for i in range(runs):
        texts = [TEXTS[(i + j) % len(TEXTS)] for j in range(batch)]
        start = time.perf_counter()
        predict_batch(sess, tokenizer, texts, max_length, stride)
        samples.append(time.perf_counter() - start)
    ordered = sorted(samples)
    mean = statistics.mean(samples)
    return {
        "runs": len(samples),
        "p50_ms": statistics.median(samples) * 1000,
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))] * 1000,
        "mean_ms": mean * 1000,
        "texts_per_s": batch / mean,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = args.threads
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"], sess_options=options)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    # Warmup.
    predict_batch(sess, tokenizer, TEXTS[:4], args.max_length, args.stride)

    report = {
        "onnx": str(args.onnx),
        "threads": args.threads,
        "max_length": args.max_length,
        "stride": args.stride,
        "scope": "CPU full windowed inference incl. tokenization + ONNX + BIO decode; no HTTP",
    }
    for batch in (1, 8, 16, 32):
        report[str(batch)] = measure(sess, tokenizer, batch, args.max_length, args.stride)
        print(f"batch={batch}: {json.dumps(report[str(batch)])}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()