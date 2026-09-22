#!/usr/bin/env python3
"""Integrate wolframko/russian-pii-66k into the ru-pii corpus layout.

Downloads the source parquet from the HuggingFace Hub, maps Presidio-style
labels onto the hackathon label set (splitting combined passport and driver
license numbers into series + number spans, merging adjacent GIVENNAME and
SURNAME into single PERSON entities), deduplicates against data/hybrid, and
writes data/wolframko/{train,dev,test}.jsonl with manifest.json.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "wolframko"
RAW_PATH = OUT_DIR / "raw" / "train-00000-of-00001.parquet"
PARQUET_URL = "https://huggingface.co/datasets/wolframko/russian-pii-66k/resolve/main/data/train-00000-of-00001.parquet"
HYBRID_DIR = REPO_ROOT / "data" / "hybrid"
SOURCE_NAME = "wolframko_russian_pii_66k"
SEED_SPLIT = 20260922

LABEL_MAP = {
    "GIVENNAME": "PERSON",
    "SURNAME": "PERSON",
    "TELEPHONENUM": "PHONE",
    "EMAIL": "EMAIL",
    "CITY": "CITY",
    "STREET": "STREET",
    "BUILDINGNUM": "HOUSE",
    "ZIPCODE": "POSTAL_CODE",
    "TAXNUM": "INN",
    "CREDITCARDNUMBER": "CARD_NUMBER",
    "DATEOFBIRTH": "BIRTH_DATE",
}
SPLIT_LABELS = {
    "IDCARDNUM": ("PASSPORT_SERIES", "PASSPORT_NUMBER"),
    "DRIVERLICENSENUM": ("DRIVER_LICENSE_SERIES", "DRIVER_LICENSE_NUMBER"),
}
DROPPED_LABELS = {"USERNAME", "PASSWORD", "SOCIALNUM", "ACCOUNTNUM"}
PERSON_PARTS = {"GIVENNAME", "SURNAME"}
COMBINED_RE = re.compile(r"^(\d{2}\s?\d{2})\s+(\d{6})$")
SPLIT_RATIOS = {"train": 0.80, "dev": 0.10, "test": 0.10}


def download_parquet() -> None:
    if RAW_PATH.exists():
        print(f"parquet already present: {RAW_PATH}")
        return
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {PARQUET_URL}")
    with urllib.request.urlopen(PARQUET_URL, timeout=180) as resp, RAW_PATH.open("wb") as dst:
        while chunk := resp.read(1 << 20):
            dst.write(chunk)
    print(f"saved {RAW_PATH.stat().st_size / 1e6:.1f} MB")


def hybrid_texts() -> set[str]:
    texts: set[str] = set()
    for split in ("train", "dev", "test"):
        path = HYBRID_DIR / f"{split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    texts.add(json.loads(line)["text"])
    return texts


def split_of(text: str) -> str:
    digest = hashlib.md5((str(SEED_SPLIT) + text).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 2**32
    if bucket < SPLIT_RATIOS["train"]:
        return "train"
    if bucket < SPLIT_RATIOS["train"] + SPLIT_RATIOS["dev"]:
        return "dev"
    return "test"


def map_entities(text: str, privacy_mask: list[dict], stats: dict) -> list[tuple]:
    out = []
    for ent in sorted(privacy_mask, key=lambda e: (e["start"], e["end"])):
        start, end = ent["start"], ent["end"]
        label, value = ent["label"], ent.get("value")
        if end <= start:
            stats["empty_spans"] += 1
            continue
        if label in DROPPED_LABELS:
            stats["dropped_labels"][label] += 1
            continue
        if label not in LABEL_MAP and label not in SPLIT_LABELS:
            stats["dropped_labels"][label] += 1
            continue
        if text[start:end] != value:
            stats["offset_mismatch"] += 1
            continue
        if label in SPLIT_LABELS:
            m = COMBINED_RE.match(value)
            if not m:
                stats["unsplit_combined"][label] += 1
                continue
            series_lab, number_lab = SPLIT_LABELS[label]
            out.append((start + m.start(1), start + m.end(1), series_lab, m.group(1), label))
            out.append((start + m.start(2), start + m.end(2), number_lab, m.group(2), label))
            continue
        out.append((start, end, LABEL_MAP[label], value, label))
    return out


def merge_person_parts(ents: list[tuple], text: str, stats: dict) -> list[tuple]:
    changed = True
    while changed:
        changed = False
        merged = []
        i = 0
        while i < len(ents):
            cur = ents[i]
            if i + 1 < len(ents):
                nxt = ents[i + 1]
                gap = text[cur[1]:nxt[0]]
                if {cur[4], nxt[4]} == PERSON_PARTS and 0 <= len(gap) <= 2 and (not gap or gap.isspace()):
                    merged.append((cur[0], nxt[1], "PERSON", text[cur[0]:nxt[1]], cur[4] + "+" + nxt[4]))
                    stats["person_merged"] += 1
                    changed = True
                    i += 2
                    continue
            merged.append(cur)
            i += 1
        ents = merged
    return ents


def drop_crossing(ents: list[tuple], stats: dict) -> list[tuple]:
    open_ends: list[int] = []
    out = []
    for ent in sorted(ents, key=lambda e: (e[0], e[1])):
        while open_ends and open_ends[-1] <= ent[0]:
            open_ends.pop()
        if open_ends and ent[1] > open_ends[-1]:
            stats["crossing_dropped"] += 1
            continue
        out.append(ent)
        open_ends.append(ent[1])
    return out


def main() -> int:
    download_parquet()
    frame = pd.read_parquet(RAW_PATH)
    print(f"source rows: {len(frame)}")
    known = hybrid_texts()
    print(f"hybrid texts loaded for dedup: {len(known)}")

    stats = {
        "dropped_labels": Counter(),
        "unsplit_combined": Counter(),
        "empty_spans": 0,
        "offset_mismatch": 0,
        "crossing_dropped": 0,
        "person_merged": 0,
        "dup_internal": 0,
        "dup_vs_hybrid": 0,
        "empty_rows": 0,
    }
    by_split: dict[str, list] = {"train": [], "dev": [], "test": []}
    seen_texts: set[str] = set()

    for row in frame.itertuples(index=False):
        text = row.source_text
        if not isinstance(text, str) or not text.strip():
            stats["empty_rows"] += 1
            continue
        if text in seen_texts:
            stats["dup_internal"] += 1
            continue
        seen_texts.add(text)
        if text in known:
            stats["dup_vs_hybrid"] += 1
            continue
        ents = map_entities(text, list(row.privacy_mask), stats)
        ents = merge_person_parts(ents, text, stats)
        ents = drop_crossing(ents, stats)
        by_split[split_of(text)].append((text, ents))

    for split, items in by_split.items():
        path = OUT_DIR / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for n, (text, ents) in enumerate(items, start=1):
                record = {
                    "id": f"wolframko-{split}-{n:06d}",
                    "text": text,
                    "split": split,
                    "source": SOURCE_NAME,
                    "annotated_labels": sorted({e[2] for e in ents}),
                    "entities": [
                        {"start": s, "end": e, "label": lab, "text": val}
                        for s, e, lab, val, _orig in ents
                    ],
                    "language": "ru",
                    "synthetic": True,
                    "provenance": "Presidio-style synthetic Russian PII texts",
                    "source_url": "https://huggingface.co/datasets/wolframko/russian-pii-66k",
                    "annotation_method": "published privacy_mask char offsets mapped to task taxonomy",
                    "license": "not specified on HF dataset card; synthetic generated data",
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": "ru-pii-wolframko-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED_SPLIT,
        "inputs": {"parquet": str(RAW_PATH), "url": PARQUET_URL},
        "split_ratios": {"train": 0.96, "dev": 0.02, "test": 0.02},
        "label_map": LABEL_MAP,
        "split_labels": {k: list(v) for k, v in SPLIT_LABELS.items()},
        "dropped_source_labels": sorted(DROPPED_LABELS),
        "audit": {
            "splits": {
                split: {
                    "records": len(items),
                    "entities": sum(len(ents) for _, ents in items),
                    "by_label": dict(sorted(Counter(
                        lab for _, ents in items for _, _, lab, _, _ in ents
                    ).items())),
                    "by_source": {SOURCE_NAME: len(items)},
                }
                for split, items in by_split.items()
            },
            "stats": {k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.items()},
            "checks_passed": True,
        },
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = f"""# wolframko/russian-pii-66k (integrated)

Source: https://huggingface.co/datasets/wolframko/russian-pii-66k
({len(frame)} synthetic Russian PII texts, Presidio-style generator, char-offset spans).

Integration (see ../../scripts/integrate_wolframko.py, manifest.json for details):

- LABEL_MAP: GIVENNAME/SURNAME -> PERSON (adjacent pairs merged into one span),
  TELEPHONENUM -> PHONE, BUILDINGNUM -> HOUSE, ZIPCODE -> POSTAL_CODE,
  TAXNUM -> INN, CREDITCARDNUMBER -> CARD_NUMBER, DATEOFBIRTH -> BIRTH_DATE,
  CITY/STREET/EMAIL kept as is.
- IDCARDNUM / DRIVERLICENSENUM ("NN NN NNNNNN") split into series + number spans.
- USERNAME, PASSWORD, SOCIALNUM, ACCOUNTNUM dropped (not in the task TZ).
- Deduplicated against data/hybrid by exact text; deterministic 80/10/10 split by md5.
- License: not specified on the HF dataset card; synthetic generated data.

Splits: {', '.join(f"{s}: {len(by_split[s])} records" for s in ('train', 'dev', 'test'))}.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    print(f"\n{'split':<6}{'records':>9}{'entities':>10}")
    for split, items in by_split.items():
        print(f"{split:<6}{len(items):>9}{sum(len(ents) for _, ents in items):>10}")
    print(f"\nlabel counts (all splits):")
    total = Counter()
    for items in by_split.values():
        total.update(lab for _, ents in items for _, _, lab, _, _ in ents)
    for lab, c in total.most_common():
        print(f"  {lab:<24}{c}")
    print(f"\nstats: {json.dumps({k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.items()}, ensure_ascii=False, indent=1)}")
    print(f"\nwritten to {OUT_DIR}")
    if stats["offset_mismatch"] or stats["crossing_dropped"]:
        print("WARNING: offset mismatches or crossing spans were dropped - review stats above", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
