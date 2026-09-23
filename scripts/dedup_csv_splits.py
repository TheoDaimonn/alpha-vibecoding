#!/usr/bin/env python3
"""Remove cross-split duplicate texts from data/csv.

A row whose normalized text already appears in an earlier split (in the order
train -> dev -> test) is dropped, so no text appears in more than one split.
Ids, row order, and all other rows are preserved.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "dev", "test")
CSV_FIELDS = ("id", "split", "source", "text", "entities")


def norm_text(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def main() -> int:
    data_dir = REPO_ROOT / "data" / "csv"
    seen: set[str] = set()
    removed = 0
    for split in SPLITS:
        path = data_dir / f"{split}.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        kept = []
        for row in rows:
            key = norm_text(row["text"])
            if key in seen:
                removed += 1
                continue
            seen.add(key)
            kept.append(row)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(kept)
        print(f"{split}: kept {len(kept)} of {len(rows)}")
    print(f"removed {removed} cross-split duplicates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
