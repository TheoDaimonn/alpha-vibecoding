#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m pip install -r requirements-data.txt

# 1) Human-written Russian corpora. Collection3 is off by default because its
# HF licence metadata is "other"; add it manually after your legal review.
"$PYTHON_BIN" scripts/build_real_corpus.py \
  --output data/real \
  --factru-backend auto \
  --skip-collection3

# 2) Task-oriented Russian PII train corpus (MIT). Its paired pii_benchmark is
# deliberately NOT fetched for training and remains held out for metrics.
"$PYTHON_BIN" scripts/fetch_redmadrobot_train.py --output data/redmadrobot

# 3) Focused synthetic supplement for classes missing from public corpora.
# Regenerate deterministically; packaged copies are only a convenience.
"$PYTHON_BIN" -m ru_pii.synthetic_supplement --output data/supplement

# 4) Merge + dedupe + hard fail unless every hackathon class has support.
"$PYTHON_BIN" scripts/build_hybrid_corpus.py \
  --real-dir data/real \
  --redmadrobot-dir data/redmadrobot \
  --supplement-dir data/supplement \
  --output data/hybrid \
  --min-train-support 250

"$PYTHON_BIN" scripts/validate_data.py \
  --data-dir data/hybrid \
  --mode hybrid \
  --min-train-support 250

echo "Hybrid corpus ready: data/hybrid/{train,dev,test}.jsonl"
cat data/hybrid/audit.json
