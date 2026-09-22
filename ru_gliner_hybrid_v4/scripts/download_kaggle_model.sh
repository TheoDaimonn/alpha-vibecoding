#!/usr/bin/env bash
set -euo pipefail

# Download model artifacts produced by the Kaggle training kernel.
# Authentication is read from KAGGLE_API_TOKEN and never written to disk.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${KAGGLE_API_TOKEN:?Set KAGGLE_API_TOKEN in the shell; it is never stored by this script}"
: "${KAGGLE_USERNAME:?Set KAGGLE_USERNAME to your Kaggle username}"

KERNEL_SLUG="${KERNEL_SLUG:-ru-gliner-pii-2xt4}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-artifacts/kaggle-download}"
EXTRACT_DIR="${EXTRACT_DIR:-artifacts/gliner-ru-pii-kaggle}"
KERNEL_ID="${KAGGLE_USERNAME}/${KERNEL_SLUG}"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ru-gliner-kaggle-output.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

command -v kaggle >/dev/null 2>&1 || {
    echo "Kaggle CLI is required. Install it with: python3 -m pip install kaggle" >&2
    exit 127
}

mkdir -p "$DOWNLOAD_DIR"
echo "Downloading output from Kaggle kernel $KERNEL_ID..."
kaggle kernels output "$KERNEL_ID" -p "$WORK_DIR" --force

ARCHIVE="$(find "$WORK_DIR" -maxdepth 1 -type f -name 'trained_model.tar.gz' -print -quit)"
if [[ -n "$ARCHIVE" ]]; then
    rm -rf "$EXTRACT_DIR"
    mkdir -p "$EXTRACT_DIR"
    tar -xzf "$ARCHIVE" -C "$EXTRACT_DIR"
    echo "Model extracted to: $EXTRACT_DIR"
else
    cp -R "$WORK_DIR/." "$DOWNLOAD_DIR/"
    echo "Kaggle output downloaded to: $DOWNLOAD_DIR"
    echo "trained_model.tar.gz was not found; inspect downloaded output."
fi