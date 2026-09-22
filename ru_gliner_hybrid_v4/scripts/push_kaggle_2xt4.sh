#!/usr/bin/env bash
set -euo pipefail

# Upload the current corpus/model as a Kaggle notebook and start it on 2 T4 GPUs.
# Authentication is read from KAGGLE_API_TOKEN and never written to disk.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${KAGGLE_API_TOKEN:?Set KAGGLE_API_TOKEN in the shell; it is never stored by this script}"
: "${KAGGLE_USERNAME:?Set KAGGLE_USERNAME to your Kaggle username}"

KERNEL_SLUG="${KERNEL_SLUG:-ru-gliner-pii-2xt4}"
KERNEL_TITLE="${KERNEL_TITLE:-Russian GLiNER PII training 2xT4}"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ru-gliner-kaggle.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

command -v kaggle >/dev/null 2>&1 || {
    echo "Kaggle CLI is required. Install it with: python3 -m pip install kaggle" >&2
    exit 127
}

copy_file() {
    mkdir -p "$WORK_DIR/$(dirname "$1")"
    cp "$ROOT_DIR/$1" "$WORK_DIR/$1"
}

copy_tree() {
    mkdir -p "$WORK_DIR/$1"
    cp -R "$ROOT_DIR/$1/." "$WORK_DIR/$1/"
}

copy_file scripts/kaggle_train.py
copy_file scripts/kaggle_train.ipynb
copy_file scripts/train.py
copy_file scripts/post_train_benchmark.py
copy_file requirements-training.txt
copy_file pyproject.toml
copy_file configs/train_small.json
copy_tree ru_pii
copy_tree data/hybrid
copy_tree data/supplement
copy_tree data/real
copy_tree models/base-small

cat > "$WORK_DIR/kernel-metadata.json" <<EOF
{
  "id": "${KAGGLE_USERNAME}/${KERNEL_SLUG}",
  "title": "${KERNEL_TITLE}",
    "code_file": "kaggle_train.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "accelerator": "Nvidia Tesla T4",
  "gpu_count": 2,
  "internet": true
}
EOF

echo "Pushing Kaggle kernel ${KAGGLE_USERNAME}/${KERNEL_SLUG} on 2xT4..."
kaggle kernels push -p "$WORK_DIR"
echo "Kernel submitted: https://www.kaggle.com/code/${KAGGLE_USERNAME}/${KERNEL_SLUG}"