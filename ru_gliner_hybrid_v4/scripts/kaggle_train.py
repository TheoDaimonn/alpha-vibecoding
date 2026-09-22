"""Kaggle notebook entrypoint for two-T4 GLiNER training."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parent.parent
# GLiNER 0.2.29's Trainer path is not safe with DataParallel when each batch
# carries dynamic label mappings. Keep the Kaggle run on one visible T4 by
# default; set KAGGLE_USE_ALL_GPUS=1 only for an explicit DDP-compatible fork.
if os.environ.get("KAGGLE_USE_ALL_GPUS") != "1":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
output = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working/artifacts/gliner-ru-pii-small"))
device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-r",
        "requirements-training.txt",
    ],
    cwd=ROOT,
    check=True,
)

command = [
    sys.executable,
    "scripts/train.py",
    "--config",
    "configs/train_small.json",
    "--model",
    "models/base-small",
    "--device",
    device,
    "--output",
    str(output),
    "--benchmark-external",
    "none",
]
if device == "cpu":
    command.append("--allow-slow-cpu")

print(f"Kaggle device: {device}", flush=True)
subprocess.run(command, cwd=ROOT, check=True)
archive = output.parent / "trained_model.tar.gz"
with tarfile.open(archive, "w:gz") as tar:
    tar.add(output, arcname=output.name)
print(f"Download archive: {archive}", flush=True)
print(f"Training artifacts: {output}", flush=True)