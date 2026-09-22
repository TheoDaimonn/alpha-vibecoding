"""Run post-training metrics on the real held-out test and Russian external benchmarks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ru_pii.benchmarking import evaluate_hivetrace, evaluate_redmadrobot, write_markdown_report
from ru_pii.inference import RussianPIIDetector
from ru_pii.metrics import span_metrics
from ru_pii.schema import read_jsonl


def eval_local(detector: RussianPIIDetector, path: str, batch_size: int) -> dict:
    rows = read_jsonl(path)
    preds = []
    texts = [r["text"] for r in rows]
    for i in range(0, len(texts), batch_size):
        batch = detector.predict_batch(texts[i:i + batch_size], include_auxiliary=True)
        preds.extend([[e.to_dict() for e in doc] for doc in batch])
    return {"status": "ok", **span_metrics(rows, preds)}


def run_suite(*, model: str, device: str, output: str, batch_size: int = 16,
              external: str = "auto", cache_dir: str | None = None) -> dict:
    detector = RussianPIIDetector.from_pretrained(model, device=device, batch_size=batch_size)
    result = {
        "model": model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "external_mode": external,
        "benchmarks": {},
    }
    local_sets = {
        "hybrid_heldout_test": "data/hybrid/test.jsonl",
        "real_heldout_test": "data/real/test.jsonl",
        "synthetic_task_coverage_test": "data/supplement/test.jsonl",
    }
    for name, path in local_sets.items():
        if Path(path).is_file():
            result["benchmarks"][name] = eval_local(detector, path, batch_size)
        else:
            result["benchmarks"][name] = {"status": "skipped", "reason": "held-out file missing; build the corpus first"}

    if external != "none":
        for name, fn in [("hivetrace_pii_bench_ru", evaluate_hivetrace), ("redmadrobot_pii_benchmark_ru", evaluate_redmadrobot)]:
            try:
                result["benchmarks"][name] = {"status": "ok", **fn(detector, batch_size=batch_size, cache_dir=cache_dir)}
            except Exception as exc:
                if external == "required":
                    raise
                result["benchmarks"][name] = {
                    "status": "skipped",
                    "reason": type(exc).__name__,
                    "note": "External benchmark unavailable; local held-out metrics were still computed.",
                }

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(result, out.with_suffix(".md"))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="reports/post_train_benchmark.json")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--external", choices=["auto", "required", "none"], default="auto",
                   help="auto=try Russian HF benchmarks and continue offline; required=fail if unavailable")
    p.add_argument("--cache-dir", default=None)
    args = p.parse_args()
    run_suite(model=args.model, device=args.device, output=args.output, batch_size=args.batch_size,
              external=args.external, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
