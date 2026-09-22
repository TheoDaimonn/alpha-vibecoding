"""Merge source corpora with the focused synthetic task supplement.

The build fails if no sourced training rows are present, or if any task-required
class has zero/minimal support.  Held-out external PII benchmarks are never read.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ru_pii.schema import TASK_REQUIRED_LABELS, read_jsonl, write_jsonl, validate_record


def key(text: str) -> str:
    return hashlib.sha256(" ".join(text.casefold().split()).encode("utf-8")).hexdigest()


def load_if(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def dedupe(splits: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    seen: set[str] = set()
    dropped = Counter()
    out = {"train": [], "dev": [], "test": []}
    # Protect held-out sets first.
    for split in ("test", "dev", "train"):
        for row in splits[split]:
            k = key(row["text"])
            if k in seen:
                dropped[split] += 1
                continue
            seen.add(k)
            out[split].append(row)
    return out, dict(dropped)


def audit(splits: dict[str, list[dict[str, Any]]], *, min_support: int, max_synthetic_fraction: float) -> dict[str, Any]:
    report: dict[str, Any] = {"splits": {}}
    for split, rows in splits.items():
        labels, sources = Counter(), Counter()
        synthetic, sourced = 0, 0
        for row in rows:
            validate_record(row)
            labels.update(e["label"] for e in row["entities"])
            sources[row.get("source", "unknown")] += 1
            if row.get("synthetic") is True:
                synthetic += 1
            else:
                sourced += 1
        task = {k: labels.get(k, 0) for k in sorted(TASK_REQUIRED_LABELS)}
        report["splits"][split] = {
            "records": len(rows), "entities": sum(labels.values()),
            "sourced_or_mixed_rows": sourced, "pure_synthetic_rows": synthetic,
            "pure_synthetic_fraction": synthetic / max(1, len(rows)),
            "by_source": dict(sorted(sources.items())), "by_label": dict(sorted(labels.items())),
            "task_required_support": task,
        }
    train = report["splits"]["train"]
    report["missing_required_train"] = [k for k, v in train["task_required_support"].items() if v < min_support]
    report["checks_passed"] = (
        not report["missing_required_train"]
        and train["sourced_or_mixed_rows"] > 0
        and train["pure_synthetic_fraction"] <= max_synthetic_fraction
    )
    report["max_allowed_pure_synthetic_fraction"] = max_synthetic_fraction
    if not report["checks_passed"]:
        raise ValueError("hybrid corpus fails task coverage/source-presence/synthetic-share policy")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-dir", default="data/real")
    p.add_argument("--redmadrobot-dir", default="data/redmadrobot")
    p.add_argument("--supplement-dir", default="data/supplement")
    p.add_argument("--output", default="data/hybrid")
    p.add_argument("--min-train-support", type=int, default=250)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--max-pure-synthetic-fraction", type=float, default=0.45,
                   help="hard guard: focused local supplement must not dominate train")
    args = p.parse_args()

    real, red, supp = Path(args.real_dir), Path(args.redmadrobot_dir), Path(args.supplement_dir)
    splits = {"train": [], "dev": [], "test": []}
    for split in splits:
        splits[split].extend(load_if(real / f"{split}.jsonl"))
        splits[split].extend(load_if(supp / f"{split}.jsonl"))
    rmr = load_if(red / "train.jsonl")
    if not rmr:
        raise FileNotFoundError("data/redmadrobot/train.jsonl missing; run scripts/fetch_redmadrobot_train.py")
    splits["train"].extend(rmr)

    # Require a non-synthetic source corpus beyond the generated supplement.
    sourced = [r for r in splits["train"] if r.get("synthetic") is not True]
    if not sourced:
        raise ValueError("refusing to build synthetic-only training corpus")

    splits, dropped = dedupe(splits)
    rng = random.Random(args.seed)
    for rows in splits.values():
        rng.shuffle(rows)

    if not 0.0 <= args.max_pure_synthetic_fraction < 1.0:
        p.error("--max-pure-synthetic-fraction must be in [0,1)")
    report = audit(splits, min_support=args.min_train_support, max_synthetic_fraction=args.max_pure_synthetic_fraction)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        write_jsonl(out / f"{split}.jsonl", rows)
    manifest = {
        "schema_version": "ru-pii-hybrid-v4",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "inputs": {"real": str(real), "redmadrobot": str(red), "supplement": str(supp)},
        "dedupe_dropped": dropped,
        "training_policy": "public/pseudonymized annotated corpora first; focused synthetic supplement closes uncovered task labels",
        "benchmarks_used_for_training": [],
        "task_required_labels": sorted(TASK_REQUIRED_LABELS),
        "audit": report,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
