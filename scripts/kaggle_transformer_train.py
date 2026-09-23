"""Kaggle entrypoint: train the transformer student (rubert-tiny2) on 2xT4.

Trains a small Russian transformer (rubert-tiny2, 29M params) for token-level
BIO tagging on the expanded corpus (gold + teacher predictions). The result is
a fast, high-quality PII detector that runs ~600 texts/sec on CPU.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

input_root = Path("/kaggle/input")
src_dirs = [d for d in input_root.rglob("src") if d.is_dir()]
if src_dirs:
    ROOT = src_dirs[0].parent
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ru_gliner_hybrid_v4"))
    print(f"Dataset root: {ROOT}", flush=True)

output = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working/artifacts/transformer-pii.pt"))
output.parent.mkdir(parents=True, exist_ok=True)

# Locate the corpus (prefer the expanded one if present, else the raw train).
expanded_candidates = sorted(input_root.rglob("*/expanded/train.jsonl"))
if expanded_candidates:
    train_path = expanded_candidates[0]
    dev_path = train_path.parent.parent / "ru_gliner_hybrid_v4" / "data" / "hybrid" / "dev.jsonl"
    print(f"Using expanded corpus: {train_path}", flush=True)
else:
    candidates = sorted(input_root.rglob("*/ru_gliner_hybrid_v4/data/hybrid/train.jsonl"))
    if not candidates:
        raise SystemExit("No training corpus found")
    train_path = candidates[0]
    dev_path = train_path.parent / "dev.jsonl"

# Install deps (transformers is preinstalled on Kaggle; ensure it's present).
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.40,<5", "torch"], check=True)

env = dict(os.environ)
env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "ru_gliner_hybrid_v4"), env.get("PYTHONPATH", "")])
command = [
    sys.executable,
    "-m",
    "src.models.transformer.train",
    "--train",
    str(train_path),
    "--dev",
    str(dev_path),
    "--out",
    str(output),
    "--epochs",
    os.environ.get("TRANSFORMER_EPOCHS", "8"),
    "--batch-size",
    os.environ.get("TRANSFORMER_BATCH_SIZE", "32"),
    "--max-len",
    os.environ.get("TRANSFORMER_MAX_LEN", "384"),
    "--lr",
    os.environ.get("TRANSFORMER_LR", "3e-5"),
    "--device",
    "cuda",
]
print("Running:", " ".join(command), flush=True)
subprocess.run(command, cwd=ROOT, env=env, check=True)

archive = output.parent / "transformer-pii.tar.gz"
with tarfile.open(archive, "w:gz") as tar:
    tar.add(output, arcname=output.name)
print(f"Download archive: {archive}", flush=True)