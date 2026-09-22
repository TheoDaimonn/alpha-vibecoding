"""Fetch and convert redmadrobot-rnd/pii_train (MIT) into canonical JSONL."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ru_pii.real_corpus import parse_redmadrobot_item, SOURCES
from ru_pii.schema import write_jsonl, sha256_file

REPO = "redmadrobot-rnd/pii_train"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/redmadrobot")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--revision", default=None, help="immutable HF revision; resolved automatically if omitted")
    p.add_argument("--offline", action="store_true")
    args = p.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    import pandas as pd

    if args.revision:
        revision = args.revision
    elif args.offline:
        p.error("--offline requires --revision")
    else:
        info = HfApi().dataset_info(REPO)
        if not info.sha:
            raise RuntimeError("could not resolve immutable dataset revision")
        revision = str(info.sha)

    path = hf_hub_download(
        REPO, "train.csv", repo_type="dataset", revision=revision,
        cache_dir=args.cache_dir, local_files_only=args.offline,
    )
    df = pd.read_csv(path)
    rows = []
    failed = Counter()
    for i, item in enumerate(df.to_dict(orient="records")):
        try:
            rows.append(parse_redmadrobot_item(item, index=i, split="train"))
        except (ValueError, TypeError, KeyError):
            failed["invalid_rows"] += 1
    if not rows:
        raise RuntimeError("no RedMadRobot rows converted")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    canonical = out / "train.jsonl"
    write_jsonl(canonical, rows)
    manifest = {
        "source": REPO,
        "source_url": SOURCES["redmadrobot"].url,
        "revision": revision,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "license": SOURCES["redmadrobot"].license_note,
        "provenance": SOURCES["redmadrobot"].provenance,
        "rows": len(rows),
        "rejected": dict(failed),
        "sha256": {"canonical": sha256_file(canonical)},
        "important": "pii_benchmark is NOT used for training and remains held out",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
