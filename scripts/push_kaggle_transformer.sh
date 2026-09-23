#!/usr/bin/env bash
set -euo pipefail

# Upload the transformer-training bundle (expanded corpus + src) and run on 2xT4.
# Authentication is read from KAGGLE_KEY or a kaggle.json in common locations.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

KAGGLE_USERNAME="${KAGGLE_USERNAME:-theodaimones888}"
if [[ -z "${KAGGLE_KEY:-}" ]]; then
  for f in "$HOME/Downloads/kaggle.json" "$HOME/.kaggle/kaggle.json"; do
    if [[ -f "$f" ]]; then
      KAGGLE_KEY="$(python3 -c "import json;print(json.load(open('$f'))['key'])")"
      break
    fi
  done
fi
: "${KAGGLE_KEY:?Set KAGGLE_KEY or provide kaggle.json in Downloads or ~/.kaggle}"

KERNEL_SLUG="${KERNEL_SLUG:-ru-pii-transformer-2xt4}"
KERNEL_TITLE="${KERNEL_TITLE:-ru-pii-transformer-2xt4}"
DATASET_SLUG="${DATASET_SLUG:-ru-pii-transformer-bundle}"
DATASET_ID="${KAGGLE_USERNAME}/${DATASET_SLUG}"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ru-pii-transformer-kaggle.XXXXXX")"
AUTH_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ru-pii-transformer-auth.XXXXXX")"
trap 'rm -rf "$WORK_DIR" "$AUTH_DIR"' EXIT

umask 077
printf '{"username":"%s","key":"%s"}\n' "$KAGGLE_USERNAME" "$KAGGLE_KEY" > "$AUTH_DIR/kaggle.json"
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

copy_file scripts/kaggle_transformer_train.py
cp "$ROOT_DIR/scripts/kaggle_transformer_train.py" "$WORK_DIR/kaggle_transformer_train.py"
copy_file requirements.txt
copy_tree src
copy_tree ru_gliner_hybrid_v4/ru_pii
copy_tree ru_gliner_hybrid_v4/data/hybrid

# Include the expanded corpus (gold + teacher predictions) if available.
if [[ -f "$ROOT_DIR/artifacts/train_expanded.jsonl" ]]; then
  mkdir -p "$WORK_DIR/dataset/expanded"
  cp "$ROOT_DIR/artifacts/train_expanded.jsonl" "$WORK_DIR/dataset/expanded/train.jsonl"
fi

cat > "$WORK_DIR/dataset/dataset-metadata.json" <<EOF
{
  "title": "${DATASET_SLUG}",
  "id": "${DATASET_ID}",
  "licenses": [{"name": "other"}],
  "subtitle": "Private code and corpus for transformer student training"
}
EOF

if kaggle datasets status "$DATASET_ID" >/dev/null 2>&1; then
    echo "Updating Kaggle dataset $DATASET_ID..."
    kaggle datasets version -p "$WORK_DIR/dataset" -m "Update transformer bundle" -r zip -q
else
    echo "Creating Kaggle dataset $DATASET_ID..."
    kaggle datasets create -p "$WORK_DIR/dataset" -r zip -q
fi

cat > "$WORK_DIR/kernel-metadata.json" <<EOF
{
  "id": "${KAGGLE_USERNAME}/${KERNEL_SLUG}",
  "title": "${KERNEL_TITLE}",
  "code_file": "kaggle_transformer_train.py",
  "language": "python",
  "kernel_type": "script",
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