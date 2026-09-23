#!/usr/bin/env python3
"""Strip famous public figures from PERSON annotations in data/hybrid.

Famous historical/cultural figures (writers, scientists, composers, painters,
pre-revolutionary statesmen, cosmonauts) are annotated as plain PERSON in
factrueval2016/nerel news corpora. Per the hackathon TZ a public figure
mention is not personal data, so those spans must not be masked. This script
removes both the original PERSON span and any PUBLIC_PERSON twin (added by
an earlier revision of this pipeline) for every span whose words match a
curated list of famous surnames (stem + case-suffix matching). It also
removes PUBLIC_ADDRESS bank-branch spans together with entities nested
inside them (bank branch addresses are not personal data either). Updates
hybrid/manifest.json counts and prints a report. Modern politicians are
deliberately not touched. Idempotent: a second run removes nothing.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_persons import relabel_stems

REPO_ROOT = Path(__file__).resolve().parents[1]
HYBRID_DIR = REPO_ROOT / "data" / "hybrid"
SPLITS = ("train", "dev", "test")
SUFFIXES = {"", "а", "у", "ю", "е", "я", "ом", "ем", "ым", "ой", "ей", "ую", "ая", "ого", "ому", "им", "ых", "ими"}
WORD_RE = re.compile(r"[а-яё]+", re.IGNORECASE)


def match_famous(text_value: str, stems: dict[str, str]) -> str | None:
    for word in WORD_RE.findall(text_value):
        low = word.lower()
        for stem, surname in stems.items():
            if low.startswith(stem) and low[len(stem):] in SUFFIXES:
                return surname
    return None


def strip_split(path: Path, stems: dict[str, str], report: dict) -> int:
    records = []
    removed = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record["text"]
            public_addr = [
                (e["start"], e["end"])
                for e in record["entities"]
                if e["label"] == "PUBLIC_ADDRESS"
            ]
            kept_entities = []
            for entity in record["entities"]:
                if entity["label"] == "PUBLIC_PERSON":
                    removed += 1
                    report[match_famous(
                        entity.get("text", text[entity["start"]:entity["end"]]), stems
                    ) or "?"] += 1
                    continue
                if entity["label"] == "PUBLIC_ADDRESS":
                    removed += 1
                    continue
                if any(ps <= entity["start"] and entity["end"] <= pe for ps, pe in public_addr):
                    removed += 1
                    continue
                if entity["label"] == "PERSON":
                    surname = match_famous(
                        entity.get("text", text[entity["start"]:entity["end"]]), stems
                    )
                    if surname is not None:
                        removed += 1
                        report[surname] += 1
                        continue
                kept_entities.append(entity)
            record["entities"] = kept_entities
            if public_addr:
                record["annotated_labels"] = sorted(
                    set(record.get("annotated_labels") or []) - {"PUBLIC_ADDRESS"}
                )
            records.append(record)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return removed


def update_manifest(stats: dict) -> None:
    manifest_path = HYBRID_DIR / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    for split in SPLITS:
        audit = manifest["audit"]["splits"][split]
        audit["entities"] = stats[split]["entities"]
        audit["by_label"] = stats[split]["by_label"]
        if "task_required_support" in audit:
            support = audit["task_required_support"]
            if "PERSON" in support:
                support["PERSON"] = stats[split]["by_label"].get("PERSON", 0)
            support.pop("PUBLIC_PERSON", None)
    manifest["audit"]["public_person_relabel"] = {
        "stripped_spans": {split: stats[split]["removed"] for split in SPLITS},
        "policy": "famous public figures and bank branch addresses are not personal data; spans removed entirely (no PUBLIC_* labels)",
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> int:
    stems = relabel_stems()
    print(f"famous surname stems: {len(stems)}")
    report: dict = Counter()
    stats = {}
    for split in SPLITS:
        path = HYBRID_DIR / f"{split}.jsonl"
        removed = strip_split(path, stems, report)
        by_label: Counter = Counter()
        entities_total = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                entities_total += len(record["entities"])
                by_label.update(e["label"] for e in record["entities"])
        stats[split] = {
            "removed": removed,
            "entities": entities_total,
            "by_label": dict(sorted(by_label.items())),
        }
        print(f"{split}: -{removed} famous-person spans, {entities_total} entities total")

    update_manifest(stats)
    print("\nby famous surname:")
    for surname, count in sorted(report.items()):
        print(f"  {surname}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
