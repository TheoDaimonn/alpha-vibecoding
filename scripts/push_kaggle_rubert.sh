#!/usr/bin/env bash
# Default: prepare only. Add --submit to upload privately and start training.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
exec "$PYTHON_BIN" "$ROOT_DIR/scripts/push_kaggle_recall.py" --model base --epochs 10 \
  --max-length 1024 --stride 256 --batch-size 2 \
  --gradient-accumulation-steps 4 --gradient-checkpointing --extend-positions \
  --run-suffix 1024-v1 "$@"
