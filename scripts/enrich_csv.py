#!/usr/bin/env python3
"""Restore bonus labels from data/hybrid into data/csv rows.

The CSV build filtered hybrid entities down to the 24 mandatory task labels,
dropping identity-document bonus labels that are worth extra hackathon points
(TZ p. 6): SNILS, OMS, MILITARY_ID, BIRTH_CERTIFICATE, plus IP_ADDRESS. This
script merges those entities back into the CSV rows built from data/hybrid
(matched by record id and verified against identical text). Famous-person
stripping is NOT redone here: data/hybrid is the already-relabelled source of
truth (scripts/relabel_public_persons.py).

Idempotent: a second run adds nothing. Rows without a hybrid match are left
untouched. Validation: offsets must match, spans must not cross.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HYBRID_DIR = REPO_ROOT / "data" / "hybrid"
CSV_DIR = REPO_ROOT / "data" / "csv"
SPLITS = ("train", "dev", "test")
CSV_FIELDS = ("id", "split", "source", "text", "entities")
BONUS_LABELS = frozenset({"SNILS", "OMS", "MILITARY_ID", "BIRTH_CERTIFICATE", "IP_ADDRESS"})


class EnrichmentError(Exception):
    """Raised when a merged entity violates validation rules."""


def load_hybrid() -> dict[str, tuple[str, list[dict]]]:
    records: dict[str, tuple[str, list[dict]]] = {}
    for split in SPLITS:
        with (HYBRID_DIR / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                records[record["id"]] = (record["text"], record["entities"])
    return records


def validate_no_crossing(text: str, entities: list[dict], location: str) -> None:
    open_ends: list[int] = []
    for e in sorted(entities, key=lambda x: (x["start"], x["start"] - x["end"])):
        if text[e["start"]:e["end"]] != e["text"]:
            raise EnrichmentError(f"{location}: offset mismatch {e!r}")
        while open_ends and open_ends[-1] <= e["start"]:
            open_ends.pop()
        if open_ends and e["end"] > open_ends[-1]:
            raise EnrichmentError(f"{location}: crossing spans at {e['text']!r}")
        open_ends.append(e["end"])


def main() -> int:
    hybrid = load_hybrid()
    added = Counter()
    matched_rows = 0
    for split in SPLITS:
        path = CSV_DIR / f"{split}.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        changed = 0
        for row in rows:
            record = hybrid.get(row["id"])
            if record is None:
                continue
            text, entities = record
            if text != row["text"]:
                print(f"warning: text mismatch for {row['id']}, skipped", file=sys.stderr)
                continue
            matched_rows += 1
            existing = json.loads(row["entities"])
            have = {(e["start"], e["end"], e["label"]) for e in existing}
            merged = list(existing)
            for e in entities:
                if e["label"] not in BONUS_LABELS:
                    continue
                key = (e["start"], e["end"], e["label"])
                if key in have:
                    continue
                merged.append({"start": e["start"], "end": e["end"], "label": e["label"], "text": text[e["start"]:e["end"]]})
                have.add(key)
                added[e["label"]] += 1
            if len(merged) != len(existing):
                merged.sort(key=lambda e: (e["start"], e["start"] - e["end"]))
                validate_no_crossing(text, merged, row["id"])
                row["entities"] = json.dumps(merged, ensure_ascii=False, separators=(",", ":"))
                changed += 1
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"{split}: {changed} rows enriched")
    print(f"hybrid-matched rows: {matched_rows}")
    print("restored entities:", dict(sorted(added.items(), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
