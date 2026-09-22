#!/usr/bin/env python3
"""Convert the hybrid PII corpus from JSONL to record-level CSV files.

Reads ``data/hybrid/{train,dev,test}.jsonl`` and writes ``data/csv/{split}.csv``
with one row per text and character-offset entities serialized as compact JSON.

Only hackathon-required PII labels (plus PUBLIC_PERSON / PUBLIC_ADDRESS
context negatives) are kept. Rows whose entities are all filtered out stay in
the output as hard negatives. Validation fails loudly on offset mismatches,
duplicate ids, split inconsistencies, or crossing (non-nested) spans.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ALLOWED_LABELS = frozenset(
    {
        "ADDRESS",
        "APARTMENT",
        "BIRTH_DATE",
        "BIRTH_PLACE",
        "CARDHOLDER",
        "CARD_NUMBER",
        "CITIZENSHIP",
        "CITY",
        "COUNTRY",
        "CVV",
        "DRIVER_LICENSE_NUMBER",
        "DRIVER_LICENSE_SERIES",
        "EMAIL",
        "HOUSE",
        "INN",
        "PASSPORT_DIVISION_CODE",
        "PASSPORT_ISSUER",
        "PASSPORT_ISSUE_DATE",
        "PASSPORT_NUMBER",
        "PASSPORT_SERIES",
        "PERSON",
        "PHONE",
        "PIN",
        "POSTAL_CODE",
        "STREET",
        "PUBLIC_ADDRESS",
        "PUBLIC_PERSON",
    }
)

SPLITS = ("train", "dev", "test")
CSV_FIELDS = ("id", "split", "source", "text", "entities")


class CorpusError(Exception):
    """Raised when the source corpus violates validation rules."""


@dataclass
class SplitStats:
    records: int = 0
    kept_entities: int = 0
    dropped_entities: int = 0
    negative_records: int = 0
    empty_text_records: int = 0
    by_label: Counter = field(default_factory=Counter)
    by_source: Counter = field(default_factory=Counter)


def iter_records(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusError(f"{path.name}:{line_no}: invalid JSON: {exc}") from exc


def filter_entities(record: dict, location: str) -> list[dict]:
    text = record.get("text")
    if not isinstance(text, str):
        raise CorpusError(f"{location}: 'text' is not a string")

    entities = record.get("entities") or []
    kept = []
    for entity in sorted(
        entities, key=lambda e: (e["start"], e["start"] - e["end"])
    ):
        start = entity["start"]
        end = entity["end"]
        label = entity["label"]
        if label not in ALLOWED_LABELS:
            continue
        span_text = text[start:end]
        if span_text != entity.get("text"):
            raise CorpusError(
                f"{location}: offset mismatch for id={record['id']!r} "
                f"label={label}: {span_text!r} != {entity.get('text')!r}"
            )
        kept.append(
            {
                "start": start,
                "end": end,
                "label": label,
                "text": span_text,
            }
        )
    return kept


def validate_nesting(kept: list[dict], location: str) -> None:
    open_ends: list[int] = []
    for entity in kept:
        start = entity["start"]
        end = entity["end"]
        while open_ends and open_ends[-1] <= start:
            open_ends.pop()
        if open_ends and end > open_ends[-1]:
            raise CorpusError(
                f"{location}: crossing entity spans for id={entity['text']!r}"
            )
        open_ends.append(end)


def convert_split(
    input_path: Path, writer: csv.DictWriter, split: str, seen_ids: set[str]
) -> SplitStats:
    stats = SplitStats()
    for line_no, record in iter_records(input_path):
        location = f"{input_path.name}:{line_no}"
        record_id = record.get("id")
        if not record_id:
            raise CorpusError(f"{location}: missing 'id'")
        if record_id in seen_ids:
            raise CorpusError(f"{location}: duplicate id {record_id!r}")
        seen_ids.add(record_id)
        if record.get("split") != split:
            raise CorpusError(
                f"{location}: split field {record.get('split')!r} != {split!r}"
            )
        if not record.get("text"):
            stats.empty_text_records += 1

        kept = filter_entities(record, location)
        validate_nesting(kept, location)
        if not kept:
            stats.negative_records += 1
        stats.records += 1
        stats.kept_entities += len(kept)
        stats.dropped_entities += len(record.get("entities") or []) - len(kept)
        stats.by_label.update(entity["label"] for entity in kept)
        stats.by_source[record.get("source", "")] += 1
        writer.writerow(
            {
                "id": record_id,
                "split": split,
                "source": record.get("source", ""),
                "text": record["text"],
                "entities": json.dumps(
                    kept, ensure_ascii=False, separators=(",", ":")
                ),
            }
        )
    return stats


def load_manifest_records(input_dir: Path) -> dict[str, int]:
    manifest_path = input_dir / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    splits = manifest["audit"]["splits"]
    return {split: splits[split]["records"] for split in SPLITS}


def print_report(stats_by_split: dict[str, SplitStats]) -> None:
    print(f"{'split':<6}{'records':>9}{'entities':>10}{'dropped':>9}{'neg.rows':>10}")
    for split in SPLITS:
        stats = stats_by_split[split]
        print(
            f"{split:<6}{stats.records:>9}{stats.kept_entities:>10}"
            f"{stats.dropped_entities:>9}{stats.negative_records:>10}"
        )

    all_labels = sorted(
        {label for stats in stats_by_split.values() for label in stats.by_label}
    )
    print(f"\n{'label':<24}{'train':>8}{'dev':>8}{'test':>8}")
    for label in all_labels:
        counts = [stats_by_split[split].by_label[label] for split in SPLITS]
        print(f"{label:<24}{counts[0]:>8}{counts[1]:>8}{counts[2]:>8}")

    all_sources = sorted(
        {source for stats in stats_by_split.values() for source in stats.by_source}
    )
    print(f"\n{'source':<34}{'train':>8}{'dev':>8}{'test':>8}")
    for source in all_sources:
        counts = [stats_by_split[split].by_source[source] for split in SPLITS]
        print(f"{source:<34}{counts[0]:>8}{counts[1]:>8}{counts[2]:>8}")


def accumulate(target: SplitStats, extra: SplitStats) -> None:
    target.records += extra.records
    target.kept_entities += extra.kept_entities
    target.dropped_entities += extra.dropped_entities
    target.negative_records += extra.negative_records
    target.empty_text_records += extra.empty_text_records
    target.by_label.update(extra.by_label)
    target.by_source.update(extra.by_source)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dirs",
        type=Path,
        nargs="+",
        default=[
            REPO_ROOT / "data" / "hybrid",
            REPO_ROOT / "data" / "wolframko",
            REPO_ROOT / "data" / "public_negatives",
        ],
        help="directories with {train,dev,test}.jsonl and manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "csv",
        help="directory to write {train,dev,test}.csv",
    )
    args = parser.parse_args()

    try:
        expected_by_dir = {d: load_manifest_records(d) for d in args.input_dirs}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        seen_ids: set[str] = set()
        stats_by_split = {split: SplitStats() for split in SPLITS}
        for split in SPLITS:
            output_path = args.output_dir / f"{split}.csv"
            with output_path.open("w", encoding="utf-8", newline="") as dst:
                writer = csv.DictWriter(dst, fieldnames=CSV_FIELDS)
                writer.writeheader()
                for input_dir in args.input_dirs:
                    stats = convert_split(input_dir / f"{split}.jsonl", writer, split, seen_ids)
                    if stats.records != expected_by_dir[input_dir][split]:
                        raise CorpusError(
                            f"{input_dir.name}/{split}: {stats.records} records != "
                            f"{expected_by_dir[input_dir][split]} in manifest"
                        )
                    accumulate(stats_by_split[split], stats)
    except (CorpusError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print_report(stats_by_split)
    print(f"\nCSV files written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
