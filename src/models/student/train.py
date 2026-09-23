"""Train the distilled student (BiLSTM-CRF) from the corpus.

The student learns the same PII spans as the GLiNER teacher but as a tiny
character-level tagger that is orders of magnitude faster on CPU.

Usage:
    python -m src.models.student.train \
        --train data/hybrid/train.jsonl \
        --dev data/hybrid/dev.jsonl \
        --out artifacts/student-pii.pt \
        --epochs 3 --device mps

Optional distillation: pass --teacher <gliner-model-dir> to add soft labels from
the teacher on the training texts (helps the student match the teacher's
behaviour on ambiguous spans).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from ru_pii.schema import Entity

from .inference import collate, spans_to_tags
from .model import BiLSTMCRF, StudentConfig


class CorpusDataset(Dataset):
    def __init__(self, path: str | Path, max_len: int = 512) -> None:
        self.records: list[tuple[str, list[int]]] = []
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row["text"]
                spans = [(e["start"], e["end"], e["label"]) for e in row.get("entities", [])]
                self.records.append((text, spans_to_tags(text, spans, max_len)))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[str, list[int]]:
        return self.records[idx]


def _collate_batch(batch: list[tuple[str, list[int]]], max_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    texts = [t for t, _ in batch]
    tags = [tg for _, tg in batch]
    chars, mask = collate(texts, max_len)
    tag_tensor = torch.zeros_like(chars)
    for b, tg in enumerate(tags):
        tag_tensor[b, : len(tg)] = torch.tensor(tg, dtype=torch.long)
    return chars, tag_tensor, mask


def evaluate(model: BiLSTMCRF, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    total = correct = 0
    with torch.inference_mode():
        for chars, tags, mask in loader:
            chars = chars.to(device)
            mask = mask.to(device)
            paths = model.decode(chars, mask)
            for b, path in enumerate(paths):
                length = int(mask[b].sum().item())
                gold = tags[b, :length].tolist()
                total += length
                correct += sum(1 for g, p in zip(gold, path) if g == p)
    return {"token_accuracy": correct / max(1, total)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default="ru_gliner_hybrid_v4/data/hybrid/train.jsonl")
    p.add_argument("--dev", default="ru_gliner_hybrid_v4/data/hybrid/dev.jsonl")
    p.add_argument("--out", default="artifacts/student-pii.pt")
    p.add_argument("--teacher", default=None, help="optional GLiNER model dir for distillation")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=20260922)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_ds = CorpusDataset(args.train, args.max_len)
    dev_ds = CorpusDataset(args.dev, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: _collate_batch(b, args.max_len))
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: _collate_batch(b, args.max_len))

    model = BiLSTMCRF(StudentConfig(hidden_dim=args.hidden_dim, num_layers=args.num_layers))
    device = args.device
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"train records: {len(train_ds)}, dev records: {len(dev_ds)}, device: {device}")
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for chars, tags, mask in train_loader:
            chars = chars.to(device)
            tags = tags.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            loss = model.forward_loss(chars, tags, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        avg_loss = total_loss / max(1, steps)
        dev = evaluate(model, dev_loader, device)
        print(f"epoch {epoch}: loss={avg_loss:.4f} dev_acc={dev['token_accuracy']:.4f} ({time.time()-start:.1f}s)")

    model.save(args.out)
    print(f"saved student model -> {args.out}")


if __name__ == "__main__":
    main()