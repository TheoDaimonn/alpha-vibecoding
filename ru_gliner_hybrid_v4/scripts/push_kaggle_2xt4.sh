#!/usr/bin/env bash
set -euo pipefail

# Upload the current corpus/model as a Kaggle notebook and start it on 2 T4 GPUs.
# Authentication is read from KAGGLE_API_TOKEN and never written to disk.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${KAGGLE_API_TOKEN:?Set KAGGLE_API_TOKEN in the shell; it is never stored by this script}"
KAGGLE_USERNAME="${KAGGLE_USERNAME:-theodaimones888}"

KERNEL_SLUG="${KERNEL_SLUG:-ru-gliner-pii-2xt4}"
KERNEL_TITLE="${KERNEL_TITLE:-ru-gliner-pii-2xt4}"
DATASET_SLUG="${DATASET_SLUG:-ru-gliner-pii-bundle}"
DATASET_ID="${KAGGLE_USERNAME}/${DATASET_SLUG}"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ru-gliner-kaggle.XXXXXX")"
AUTH_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ru-gliner-kaggle-auth.XXXXXX")"
trap 'rm -rf "$WORK_DIR" "$AUTH_DIR"' EXIT

# Support older Kaggle CLI versions which only read legacy kaggle.json.
umask 077
printf '{"username":"%s","key":"%s"}\n' "$KAGGLE_USERNAME" "$KAGGLE_API_TOKEN" > "$AUTH_DIR/kaggle.json"
export KAGGLE_CONFIG_DIR="$AUTH_DIR"

command -v kaggle >/dev/null 2>&1 || {
    echo "Kaggle CLI is required. Install it with: python3 -m pip install kaggle" >&2
    exit 127
}

copy_file() {
  mkdir -p "$WORK_DIR/dataset/$(dirname "$1")"
  cp "$ROOT_DIR/$1" "$WORK_DIR/dataset/$1"
}

copy_tree() {
  mkdir -p "$WORK_DIR/dataset/$1"
  cp -R "$ROOT_DIR/$1/." "$WORK_DIR/dataset/$1/"
}

copy_file scripts/kaggle_train.py
cp "$ROOT_DIR/scripts/kaggle_train.ipynb" "$WORK_DIR/kaggle_train.ipynb"
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

cat > "$WORK_DIR/dataset/dataset-metadata.json" <<EOF
{
  "title": "${DATASET_SLUG}",
  "id": "${DATASET_ID}",
  "licenses": [{"name": "other"}],
  "subtitle": "Private code, corpus and base model bundle for GLiNER training"
}
EOF

if kaggle datasets status "$DATASET_ID" >/dev/null 2>&1; then
    echo "Updating Kaggle dataset $DATASET_ID..."
    kaggle datasets version -p "$WORK_DIR/dataset" \
        -m "Update training bundle" -r zip -q
else
    echo "Creating Kaggle dataset $DATASET_ID..."
    kaggle datasets create -p "$WORK_DIR/dataset" -r zip -q
fi

cat > "$WORK_DIR/kernel-metadata.json" <<EOF
{
  "id": "${KAGGLE_USERNAME}/${KERNEL_SLUG}",
  "title": "${KERNEL_TITLE}",
    "code_file": "kaggle_train.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": ["${DATASET_ID}"],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
EOF

echo "Pushing Kaggle kernel ${KAGGLE_USERNAME}/${KERNEL_SLUG} on 2xT4..."
kaggle kernels push -p "$WORK_DIR"
echo "Kernel submitted: https://www.kaggle.com/code/${KAGGLE_USERNAME}/${KERNEL_SLUG}"