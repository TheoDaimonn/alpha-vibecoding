#!/usr/bin/env python3
"""Integrate new Russian PII datasets into data/csv.

Sources:
- hivetrace/pii-bench (HF, Apache-2.0): char-offset spans, mapped to the
  canonical schema; PASSPORT_NUMBER values are split into series/number.
- redmadrobot-rnd/pii_benchmark (HF, MIT): token/BIO rows, aligned back to the
  raw text to recover char offsets; adjacent FIRST/MIDDLE/LAST name tokens are
  merged into PERSON; document numbers are split into series/number.
- Meddies/meddies-pii config "russian" (HF, CC-BY-NC-4.0): marked-up text plus
  a clean "raw" field; spans are recomputed on the raw text.

Only ALLOWED_LABELS (the hackathon task labels plus PUBLIC_*) survive, matching
scripts/jsonl_to_csv.py conventions. Rows whose entities are fully filtered
stay as hard negatives.

Deduplication is case- and whitespace-insensitive over the full text against
the existing data/csv rows and across/inside the new sources, so a text that
already appears in any existing split is never re-added (prevents train/test
leakage when a dataset was used before). New splits are assigned
deterministically from the md5 of the normalized text: 80/10/10 train/dev/test.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from hashlib import md5
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ALLOWED_LABELS = frozenset(
    {
        "ADDRESS", "APARTMENT", "BIRTH_DATE", "BIRTH_PLACE", "CARDHOLDER",
        "CARD_NUMBER", "CITIZENSHIP", "CITY", "COUNTRY", "CVV",
        "DRIVER_LICENSE_NUMBER", "DRIVER_LICENSE_SERIES", "EMAIL", "HOUSE",
        "INN", "PASSPORT_DIVISION_CODE", "PASSPORT_ISSUER",
        "PASSPORT_ISSUE_DATE", "PASSPORT_NUMBER", "PASSPORT_SERIES",
        "PERSON", "PHONE", "PIN", "POSTAL_CODE", "STREET",
        "PUBLIC_ADDRESS", "PUBLIC_PERSON",
    }
)
NAME_PART_LABELS = frozenset({"FIRST_NAME", "MIDDLE_NAME", "LAST_NAME"})
SPLITS = ("train", "dev", "test")
CSV_FIELDS = ("id", "split", "source", "text", "entities")
NEW_SOURCES = frozenset({"hivetrace_pii_bench", "redmadrobot_pii_benchmark", "meddies_pii_ru"})
MAX_SPAN_LEN = {
    "PERSON": 100, "PHONE": 30, "EMAIL": 60, "ADDRESS": 200, "CARD_NUMBER": 30,
    "INN": 15, "CVV": 6, "CITY": 100, "COUNTRY": 60, "STREET": 150, "HOUSE": 40,
    "PASSPORT_SERIES": 12, "PASSPORT_NUMBER": 12,
    "DRIVER_LICENSE_SERIES": 12, "DRIVER_LICENSE_NUMBER": 12,
}
CYRILLIC = re.compile(r"[а-яёА-ЯЁ]")


class IntegrationError(Exception):
    """Raised when a source row violates validation rules."""


def norm_text(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def is_russian(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    cyr = sum(1 for ch in letters if CYRILLIC.match(ch))
    return cyr / len(letters) >= 0.5


def sanitize(entities: list[dict], text: str) -> list[dict]:
    """Drop implausibly long spans (corrupted markup) from a row."""
    return [e for e in entities if e["end"] - e["start"] <= MAX_SPAN_LEN.get(e["label"], 10**6)]


def split_for(key: str) -> str:
    bucket = int(md5(key.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "dev"
    return "test"


def make_entity(start: int, end: int, label: str, text: str) -> dict:
    span_text = text[start:end]
    return {"start": start, "end": end, "label": label, "text": span_text}


def validate_row(text: str, entities: list[dict], location: str) -> None:
    open_ends: list[int] = []
    for e in sorted(entities, key=lambda x: (x["start"], x["start"] - x["end"])):
        if text[e["start"]:e["end"]] != e["text"]:
            raise IntegrationError(f"{location}: offset mismatch {e!r}")
        if e["label"] not in ALLOWED_LABELS:
            raise IntegrationError(f"{location}: label {e['label']!r} not allowed")
        while open_ends and open_ends[-1] <= e["start"]:
            open_ends.pop()
        if open_ends and e["end"] > open_ends[-1]:
            raise IntegrationError(f"{location}: crossing spans at {e['text']!r}")
        open_ends.append(e["end"])


def split_document_number(start: int, end: int, text: str) -> list[dict]:
    """Split a coarse passport/driver-licence value into fine spans.

    Recognizes "NNNN NNNNNN", "NNNN-NNNNNN", 10 consecutive digits, a lone
    4-digit series, or a lone 6-digit number. Anything else (spelled-out
    digits, fragments) yields no spans.
    """
    value = text[start:end]
    spans: list[dict] = []
    pairs = re.search(r"(\d{4})\D+(\d{6})", value)
    if pairs:
        s1, s2 = pairs.span(1), pairs.span(2)
        spans.append(make_entity(start + s1[0], start + s1[1], "SERIES", text))
        spans.append(make_entity(start + s2[0], start + s2[1], "NUMBER", text))
        return spans
    ten = re.search(r"\d{10}", value)
    if ten:
        s = ten.span()
        spans.append(make_entity(start + s[0], start + s[0] + 4, "SERIES", text))
        spans.append(make_entity(start + s[0] + 4, start + s[1], "NUMBER", text))
        return spans
    four = re.search(r"\d{4}", value)
    six = re.search(r"\d{6}", value)
    if four and not six:
        s = four.span()
        spans.append(make_entity(start + s[0], start + s[1], "SERIES", text))
    elif six:
        s = six.span()
        spans.append(make_entity(start + s[0], start + s[1], "NUMBER", text))
    return spans


def extract_hivetrace() -> list[dict]:
    from datasets import load_dataset

    label_map = {
        "NAME": "PERSON",
        "PHONE_NUMBER": "PHONE",
        "EMAIL": "EMAIL",
        "BANK_CARD_NUMBER": "CARD_NUMBER",
        "ADDRESS": "ADDRESS",
        "INN": "INN",
        "CVC": "CVV",
    }
    rows: list[dict] = []
    ds = load_dataset("hivetrace/pii-bench")
    for split in sorted(ds):
        for i, ex in enumerate(ds[split]):
            location = f"hivetrace/pii-bench/{split}[{i}]"
            entities = ex["entities"]
            if isinstance(entities, str):
                entities = eval(entities, {"__builtins__": {}}, {})
            kept: list[dict] = []
            for e in entities:
                label = e.get("type")
                if label == "PASSPORT_NUMBER":
                    for ent in split_document_number(e["start"], e["end"], ex["text"]):
                        ent["label"] = f"PASSPORT_{ent['label']}"
                        kept.append(ent)
                elif label in label_map:
                    kept.append(make_entity(e["start"], e["end"], label_map[label], ex["text"]))
            validate_row(ex["text"], kept, location)
            rows.append({"source": "hivetrace_pii_bench", "text": ex["text"], "entities": kept})
    return rows


def align_tokens(text: str, tokens: list[str], location: str) -> list[int]:
    """Return the char start offset of every token in the text."""
    lowered = text.lower()
    positions: list[int] = []
    pos = 0
    for tok in tokens:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            raise IntegrationError(f"{location}: text exhausted before token {tok!r}")
        if text[pos : pos + len(tok)].lower() != tok.lower():
            idx = lowered.find(tok.lower(), pos)
            if idx < 0:
                raise IntegrationError(f"{location}: token {tok!r} not found at {pos}")
            pos = idx
        positions.append(pos)
        pos += len(tok)
    return positions


def extract_redmadrobot_benchmark() -> list[dict]:
    from datasets import load_dataset

    label_map = {
        "STREET": "STREET", "HOUSE": "HOUSE", "CITY": "CITY",
        "COUNTRY": "COUNTRY", "PHONE": "PHONE", "INN": "INN",
        "EMAIL": "EMAIL", "CREDIT_CARD": "CARD_NUMBER",
    }
    rows: list[dict] = []
    ds = load_dataset("redmadrobot-rnd/pii_benchmark")["test"]
    for i, ex in enumerate(ds):
        location = f"redmadrobot-rnd/pii_benchmark[{i}]"
        text = ex["text"]
        tokens = ex["tokens"]
        tags = ex["ner_tags"]
        if isinstance(tokens, str):
            tokens = eval(tokens, {"__builtins__": {}}, {})
        if isinstance(tags, str):
            tags = eval(tags, {"__builtins__": {}}, {})
        pairs = [(t, g) for t, g in zip(tokens, tags) if t.strip()]
        tokens = [t for t, _ in pairs]
        tags = [g for _, g in pairs]
        positions = align_tokens(text, tokens, location)
        token_labels: list[str | None] = []
        for tag in tags:
            prefix, _, label = tag.partition("-")
            token_labels.append(label if prefix in ("B", "I") and label else None)
        runs: list[tuple[int, int, str]] = []
        for j, label in enumerate(token_labels):
            if label is None:
                continue
            if runs and token_labels[j - 1] is not None and token_labels[j - 1] == label:
                s, e, lab = runs[-1]
                runs[-1] = (s, positions[j] + len(tokens[j]), lab)
            else:
                runs.append((positions[j], positions[j] + len(tokens[j]), label))
        merged: list[tuple[int, int, str]] = []
        for s, e, label in runs:
            if (
                label in NAME_PART_LABELS
                and merged
                and merged[-1][2] == "PERSON"
                and s - merged[-1][1] <= 2
            ):
                prev_s, _, _ = merged[-1]
                merged[-1] = (prev_s, e, "PERSON")
            elif label in NAME_PART_LABELS:
                merged.append((s, e, "PERSON"))
            else:
                merged.append((s, e, label))
        kept: list[dict] = []
        for s, e, label in merged:
            if label == "PERSON":
                kept.append(make_entity(s, e, "PERSON", text))
            elif label in ("PASSPORT", "DRIVER_LICENSE"):
                prefix_label = "PASSPORT" if label == "PASSPORT" else "DRIVER_LICENSE"
                for ent in split_document_number(s, e, text):
                    ent["label"] = f"{prefix_label}_{ent['label']}"
                    kept.append(ent)
            elif label in label_map:
                kept.append(make_entity(s, e, label_map[label], text))
        kept = sanitize(kept, text)
        validate_row(text, kept, location)
        rows.append({"source": "redmadrobot_pii_benchmark", "text": text, "entities": kept})
    return rows


MEDDIES_MARKER = re.compile(r"\[([^\[\]]*)\]<([a-z_]+)>")
MEDDIES_LABEL_MAP = {
    "human_name": "PERSON",
    "phone_number": "PHONE",
    "email_address": "EMAIL",
    "address": "ADDRESS",
}


def extract_meddies_russian() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("Meddies/meddies-pii", "russian")["train"]
    rows: list[dict] = []
    failures = 0
    for i, ex in enumerate(ds):
        location = f"Meddies/meddies-pii russian[{i}]"
        marked, raw = ex["text"], ex["raw"]
        entities: list[dict] = []
        cursor = 0
        ok = True
        for m in MEDDIES_MARKER.finditer(marked):
            value, category = m.group(1), m.group(2)
            label = MEDDIES_LABEL_MAP.get(category)
            if not value:
                continue
            idx = raw.find(value, cursor)
            if idx < 0:
                if label:
                    ok = False
                    break
                continue
            cursor = idx + len(value)
            if label:
                entities.append(make_entity(idx, idx + len(value), label, raw))
        if not ok:
            failures += 1
            continue
        entities.sort(key=lambda e: (e["start"], e["start"] - e["end"]))
        dedup: list[dict] = []
        for e in entities:
            if not any(d["start"] == e["start"] and d["end"] == e["end"] and d["label"] == e["label"] for d in dedup):
                dedup.append(e)
        try:
            validate_row(raw, dedup, location)
        except IntegrationError:
            failures += 1
            continue
        rows.append({"source": "meddies_pii_ru", "text": raw, "entities": dedup})
    if failures:
        print(f"meddies: skipped {failures} rows failing markup/raw reconstruction", file=sys.stderr)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "csv")
    parser.add_argument("--dry-run", action="store_true", help="only report, do not rewrite CSVs")
    args = parser.parse_args()

    existing: dict[str, list[dict]] = {s: [] for s in SPLITS}
    seen: dict[str, str] = {}
    for split in SPLITS:
        with (args.output_dir / f"{split}.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["source"] in NEW_SOURCES:
                    continue
                row["entities"] = json.loads(row["entities"])
                existing[split].append(row)
                seen.setdefault(norm_text(row["text"]), split)

    extractors = [
        ("hivetrace/pii-bench", extract_hivetrace),
        ("redmadrobot-rnd/pii_benchmark", extract_redmadrobot_benchmark),
        ("Meddies/meddies-pii russian", extract_meddies_russian),
    ]
    stats = {name: {"rows": 0, "dups_existing": 0, "non_russian": 0} for name, _ in extractors}
    additions: dict[str, list[dict]] = {s: [] for s in SPLITS}
    label_counts: Counter = Counter()
    for name, extractor in extractors:
        for row in extractor():
            if not is_russian(row["text"]):
                stats[name]["non_russian"] += 1
                continue
            row["entities"] = sanitize(row["entities"], row["text"])
            validate_row(row["text"], row["entities"], name)
            key = norm_text(row["text"])
            if key in seen:
                stats[name]["dups_existing"] += 1
                continue
            seen[key] = "new"
            split = split_for(key)
            stats[name]["rows"] += 1
            label_counts.update(e["label"] for e in row["entities"])
            prefix = {"hivetrace_pii_bench": "hivetrace", "redmadrobot_pii_benchmark": "rmr-bench", "meddies_pii_ru": "meddies-ru"}[row["source"]]
            additions[split].append(
                {
                    "id": f"{prefix}-{split}-{len(additions[split])}",
                    "split": split,
                    "source": row["source"],
                    "text": row["text"],
                    "entities": row["entities"],
                }
            )

    print(f"{'source':<28}{'added':>8}{'dups':>8}{'non-ru':>8}")
    for name, _ in extractors:
        print(
            f"{name:<28}{stats[name]['rows']:>8}{stats[name]['dups_existing']:>8}"
            f"{stats[name]['non_russian']:>8}"
        )
    print(f"\n{'split':<8}{'old':>8}{'new':>8}{'total':>8}")
    grand_new = 0
    for split in SPLITS:
        print(f"{split:<8}{len(existing[split]):>8}{len(additions[split]):>8}{len(existing[split]) + len(additions[split]):>8}")
        grand_new += len(additions[split])
    print("\nnew label counts:", dict(sorted(label_counts.items(), key=lambda kv: -kv[1])))
    print("new texts total:", grand_new)

    if args.dry_run:
        return 0

    for split in SPLITS:
        path = args.output_dir / f"{split}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in existing[split] + additions[split]:
                out = dict(row)
                out["entities"] = json.dumps(out["entities"], ensure_ascii=False, separators=(",", ":"))
                writer.writerow(out)
        print(f"wrote {path} ({len(existing[split]) + len(additions[split])} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
