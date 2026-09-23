"""Train the transformer student (rubert-tiny2) for token-level BIO tagging.

Usage:
    python -m src.models.transformer.train \
        --train data/hybrid/train.jsonl \
        --dev data/hybrid/dev.jsonl \
        --out artifacts/transformer-pii.pt \
        --epochs 5 --device cuda

The corpus has char-level spans; they are converted to token-level BIO tags via
the tokenizer's offset mapping.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .model import TransformerNER, TransformerConfig, tag_id, O_TAG


class CorpusDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer, max_len: int = 256) -> None:
        self.records: list[tuple[list[int], list[int], list[int], list[float]]] = []  # (input_ids, attn, labels, weights)
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row["text"]
                spans = [(e["start"], e["end"], e["label"], float(e.get("weight", 1.0))) for e in row.get("entities", [])]
                self.records.append(self._encode(text, spans, tokenizer, max_len))

    def _encode(self, text: str, spans, tokenizer, max_len: int):
        enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_len)
        input_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        labels = [O_TAG] * len(input_ids)
        weights = [1.0] * len(input_ids)
        # Sort spans by start so we can assign B/I correctly per label.
        spans = sorted(spans, key=lambda sp: (sp[0], sp[1]))
        for start, end, label, weight in spans:
            for i, (s, e) in enumerate(offsets):
                if s == e:
                    continue
                if s >= start and e <= end:
                    # B if the previous char is NOT inside a span of the SAME label.
                    prev_in = any(ps <= s - 1 < pe and pl == label for ps, pe, pl, _ in spans)
                    labels[i] = tag_id(label, "I" if prev_in else "B")
                    weights[i] = weight
        return input_ids, [1] * len(input_ids), labels, weights

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        return self.records[idx]


def _collate(batch, pad_id: int):
    input_ids, attn, labels, weights = zip(*batch)
    max_len = max(len(x) for x in input_ids)
    pad = lambda seq: seq + [pad_id] * (max_len - len(seq))
    return (
        torch.tensor([pad(x) for x in input_ids], dtype=torch.long),
        torch.tensor([pad(x) for x in attn], dtype=torch.long),
        torch.tensor([pad(x) for x in labels], dtype=torch.long),
        torch.tensor([pad(x) for x in weights], dtype=torch.float),
    )


def evaluate(model: TransformerNER, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for input_ids, attn, labels, _weights in loader:
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            labels = labels.to(device)
            logits = model(input_ids, attn)
            preds = logits.argmax(dim=-1)
            mask = attn.bool()
            correct += ((preds == labels) & mask).sum().item()
            total += mask.sum().item()
    return {"token_accuracy": correct / max(1, total)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default="ru_gliner_hybrid_v4/data/hybrid/train.jsonl")
    p.add_argument("--dev", default="ru_gliner_hybrid_v4/data/hybrid/dev.jsonl")
    p.add_argument("--out", default="artifacts/transformer-pii.pt")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-len", type=int, default=384)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260922)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    train_ds = CorpusDataset(args.train, tokenizer, args.max_len)
    dev_ds = CorpusDataset(args.dev, tokenizer, args.max_len)
    pad_id = tokenizer.pad_token_id or 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: _collate(b, pad_id))
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: _collate(b, pad_id))

    model = TransformerNER(TransformerConfig(max_len=args.max_len))
    device = args.device
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"train records: {len(train_ds)}, dev records: {len(dev_ds)}, device: {device}")
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for input_ids, attn, labels, weights in train_loader:
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attn)
            # Plain cross-entropy over tokens (ignore padding). Weighted CE with
            # teacher-confidence weights suppressed B-tag learning; plain CE
            # lets the model learn span starts correctly.
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=pad_id
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        dev = evaluate(model, dev_loader, device)
        print(f"epoch {epoch}: loss={total_loss/max(1,steps):.4f} dev_acc={dev['token_accuracy']:.4f} ({time.time()-start:.1f}s)")

    model.save(args.out)
    print(f"saved transformer model -> {args.out}")


if __name__ == "__main__":
    main()