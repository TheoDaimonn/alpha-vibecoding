#!/usr/bin/env python3
"""Add PUBLIC_PERSON twins to classic public figures in data/hybrid.

Famous historical/cultural figures (writers, scientists, composers, painters,
pre-revolutionary statesmen, cosmonauts) are annotated as plain PERSON in
factrueval2016/nerel news corpora, which contradicts the hackathon TZ (a
public figure mention is not personal data). This script adds a PUBLIC_PERSON
twin span for every PERSON span whose words match a curated list of famous
surnames (stem + case-suffix matching), updates hybrid/manifest.json counts,
and prints a report. Modern politicians are deliberately not touched.
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


def relabel_split(path: Path, stems: dict[str, str], report: dict) -> int:
    records = []
    added = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record["text"]
            entities = record["entities"]
            has_public = {(e["start"], e["end"]) for e in entities if e["label"] == "PUBLIC_PERSON"}
            for entity in entities:
                if entity["label"] != "PERSON":
                    continue
                if (entity["start"], entity["end"]) in has_public:
                    continue
                surname = match_famous(entity.get("text", text[entity["start"]:entity["end"]]), stems)
                if surname is None:
                    continue
                entities.append({
                    "start": entity["start"],
                    "end": entity["end"],
                    "label": "PUBLIC_PERSON",
                    "text": entity["text"],
                })
                has_public.add((entity["start"], entity["end"]))
                added += 1
                report[surname] += 1
                if added <= 10:
                    report.setdefault("examples", []).append(
                        (path.name, text[entity["start"]:entity["end"]], text[:100])
                    )
            entities.sort(key=lambda e: (e["start"], e["end"]))
            records.append(record)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return added


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
            support["PUBLIC_PERSON"] = stats[split]["by_label"].get("PUBLIC_PERSON", 0)
    manifest["audit"]["public_person_relabel"] = {
        "added_twins": {split: stats[split]["added"] for split in SPLITS},
        "policy": "classic public figures only; modern politicians untouched",
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
        added = relabel_split(path, stems, report)
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
            "added": added,
            "entities": entities_total,
            "by_label": dict(sorted(by_label.items())),
        }
        print(f"{split}: +{added} PUBLIC_PERSON twins, {entities_total} entities total")

    update_manifest(stats)
    print("\nby famous surname:")
    for surname, count in sorted(report.items()):
        if surname != "examples":
            print(f"  {surname}: {count}")
    if "examples" in report:
        print("\nfirst examples:")
        for name, span, ctx in report["examples"][:5]:
            print(f"  [{name}] {span!r} in: {ctx}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
