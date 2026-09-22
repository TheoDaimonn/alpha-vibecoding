"""Offline corpus audit; checks integrity/coverage, not model quality."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ru_pii.schema import TASK_REQUIRED_LABELS, read_jsonl, corpus_audit, sha256_file
from ru_pii.real_corpus import audit_real_corpus


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/hybrid")
    p.add_argument("--output", default="reports/data_validation.json")
    p.add_argument("--mode", choices=["hybrid", "real"], default="hybrid")
    p.add_argument("--min-train-support", type=int, default=250)
    args = p.parse_args()
    base = Path(args.data_dir)
    files = {s: base / f"{s}.jsonl" for s in ["train", "dev", "test"]}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing corpus files: {missing}")
    splits = {s: read_jsonl(path) for s, path in files.items()}
    audit = corpus_audit(splits)
    train_counts = Counter(e["label"] for r in splits["train"] for e in r["entities"])
    audit["task_required_train_support"] = {k: train_counts.get(k, 0) for k in sorted(TASK_REQUIRED_LABELS)}
    audit["task_required_missing_or_weak"] = [k for k, v in audit["task_required_train_support"].items() if v < args.min_train_support]
    if audit["task_required_missing_or_weak"]:
        raise SystemExit("Required task classes are missing/weak: " + ", ".join(audit["task_required_missing_or_weak"]))
    if args.mode == "real":
        audit["real_only"] = audit_real_corpus(splits)
        if audit["real_only"]["synthetic_rows"] != 0:
            raise SystemExit("Refusing real-only corpus: synthetic rows detected")
    else:
        pure_synth = sum(r.get("synthetic") is True for r in splits["train"])
        sourced = len(splits["train"]) - pure_synth
        audit["hybrid"] = {
            "train_sourced_or_mixed_rows": sourced,
            "train_pure_synthetic_rows": pure_synth,
            "train_pure_synthetic_fraction": pure_synth / max(1, len(splits["train"])),
        }
        if sourced <= 0:
            raise SystemExit("Refusing synthetic-only training corpus")
    audit["sha256"] = {str(path): sha256_file(path) for path in files.values()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({s: d["records"] for s, d in audit["splits"].items()}, ensure_ascii=False))
    print("Offsets/labels/splits/task coverage: OK. Model quality: NOT EVALUATED.")


if __name__ == "__main__":
    main()
