#!/usr/bin/env bash
set -euo pipefail

# Run the complete workflow from TRAINING.md from any working directory.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage: bash scripts/run_training.sh

Environment overrides:
  DEVICE=cuda|cpu|auto                 (default: auto)
  BENCHMARK_EXTERNAL=auto|required|none (default: auto)
  HOST_PYTHON=/path/to/python3.13      (default: auto-detect)
  VENV_DIR=/path/to/venv               (default: .venv, or .venv-py313)
  MODEL_DIR=...                         (default: models/base-small)
  OUTPUT_DIR=...                        (default: artifacts/gliner-ru-pii-small)
EOF
    exit 0
fi

VENV_DIR_WAS_SET="${VENV_DIR+x}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
DEVICE="${DEVICE:-auto}"
BENCHMARK_EXTERNAL="${BENCHMARK_EXTERNAL:-auto}"
MODEL_DIR="${MODEL_DIR:-models/base-small}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/gliner-ru-pii-small}"

python_supported() {
    "$1" -c 'import sys; raise SystemExit(not (sys.version_info >= (3, 10) and sys.version_info < (3, 14)))' >/dev/null 2>&1
}

if [[ -x "$PYTHON_BIN" ]] && ! python_supported "$PYTHON_BIN"; then
    echo "Unsupported Python in $VENV_DIR: require Python 3.10-3.13." >&2
    PYTHON_BIN=""
    if [[ -z "$VENV_DIR_WAS_SET" ]]; then
        VENV_DIR="$ROOT_DIR/.venv-py313"
    fi
fi

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
    HOST_PYTHON="${HOST_PYTHON:-}"
    if [[ -z "$HOST_PYTHON" ]]; then
        for candidate in python3.13 python3.12 python3.11 python3; do
            if command -v "$candidate" >/dev/null 2>&1 && python_supported "$(command -v "$candidate")"; then
                HOST_PYTHON="$(command -v "$candidate")"
                break
            fi
        done
    fi
    [[ -n "$HOST_PYTHON" ]] && python_supported "$HOST_PYTHON" || {
        echo "Python 3.10-3.13 is required; set HOST_PYTHON to its executable." >&2
        exit 127
    }
    "$HOST_PYTHON" -m venv "$VENV_DIR"
    PYTHON_BIN="$VENV_DIR/bin/python"
fi

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" -m pip install -U pip
"$PYTHON_BIN" -m pip install -r requirements-data.txt
"$PYTHON_BIN" -m pip install -r requirements-training.txt

echo "== Building hybrid corpus =="
PYTHON_BIN="$PYTHON_BIN" bash scripts/make_corpus.sh

echo "== Downloading base model =="
if [[ ! -f "$MODEL_DIR/BASE_MODEL_MANIFEST.json" ]]; then
    "$PYTHON_BIN" scripts/download_model.py \
        --model gliner-community/gliner_small-v2.5 \
        --revision f227d3cd637bd4e6757ae143935316d062393341 \
        --output "$MODEL_DIR"
else
    echo "Base model already present: $MODEL_DIR"
fi

if [[ "$DEVICE" == "auto" ]]; then
    DEVICE="$($PYTHON_BIN -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
fi
TRAIN_ARGS=(--config configs/train_small.json --model "$MODEL_DIR" --device "$DEVICE")
if [[ "$DEVICE" == "cpu" ]]; then
    TRAIN_ARGS+=(--allow-slow-cpu)
fi
echo "Selected device: $DEVICE"

reset_workdir() {
    local workdir="$1"
    if [[ -e "$workdir" ]]; then
        echo "Resetting generated workdir: $workdir"
        rm -rf "$workdir"
    fi
}

if [[ -e "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Final output already exists: $OUTPUT_DIR" >&2
    echo "Set OUTPUT_DIR to a new path or remove this completed output explicitly." >&2
    exit 2
fi

echo "== Preflight =="
reset_workdir artifacts/preflight-small
"$PYTHON_BIN" scripts/train.py "${TRAIN_ARGS[@]}" \
    --prepare-only --output artifacts/preflight-small

echo "== Smoke train =="
reset_workdir artifacts/smoke-small
"$PYTHON_BIN" scripts/train.py "${TRAIN_ARGS[@]}" \
    --smoke --output artifacts/smoke-small

echo "== Full train and benchmark =="
"$PYTHON_BIN" scripts/train.py "${TRAIN_ARGS[@]}" \
    --output "$OUTPUT_DIR" --benchmark-external "$BENCHMARK_EXTERNAL"

echo "== Re-running benchmarks from exported model =="
"$PYTHON_BIN" scripts/post_train_benchmark.py \
    --model "$OUTPUT_DIR" --device "$DEVICE" \
    --external "$BENCHMARK_EXTERNAL" --output reports/rebenchmark.json

echo "Training workflow complete."
echo "Model: $OUTPUT_DIR"
echo "Status: $OUTPUT_DIR/RUN_STATUS.json"