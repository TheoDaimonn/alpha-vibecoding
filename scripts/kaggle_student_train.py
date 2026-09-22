"""Kaggle notebook entrypoint for training the distilled student on 2xT4.

Runs the char-level BiLSTM-CRF student training on GPU. The corpus is expected
under /kaggle/input/<dataset>/data/hybrid/. The trained checkpoint is written to
/kaggle/working/artifacts/student-pii.pt and packed into a tarball for download.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The dataset (with src/ and data/) is mounted under /kaggle/input/.
input_root = Path("/kaggle/input")
# Find the directory that contains src/ (the dataset root).
src_dirs = [d for d in input_root.rglob("src") if d.is_dir()]
if src_dirs:
    ROOT = src_dirs[0].parent
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ru_gliner_hybrid_v4"))
    print(f"Dataset root: {ROOT}", flush=True)
else:
    # Fallback: add every input dir to the path so imports resolve.
    for d in input_root.rglob("*"):
        if d.is_dir():
            sys.path.insert(0, str(d))
            sys.path.insert(0, str(d / "ru_gliner_hybrid_v4"))

# Use all visible GPUs (2xT4) via DataParallel for the student (it is a simple
# BiLSTM-CRF, safe to parallelise across the batch dimension).
os.environ.setdefault("KAGGLE_USE_ALL_GPUS", "1")

output = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working/artifacts/student-pii.pt"))
output.parent.mkdir(parents=True, exist_ok=True)

# Locate the corpus inside the mounted Kaggle dataset.
print("Kaggle input contents:", flush=True)
for p in sorted(input_root.rglob("train.jsonl")):
    print("  found:", p, flush=True)

candidates = sorted(input_root.rglob("*/data/hybrid/train.jsonl"))
if not candidates:
    candidates = sorted(input_root.rglob("train.jsonl"))
if not candidates:
    raise SystemExit("No training corpus found under /kaggle/input/")
train_path = candidates[0]
dev_path = train_path.parent / "dev.jsonl"
print(f"Using train: {train_path}", flush=True)

# torch is preinstalled on Kaggle GPU kernels; ensure numpy is present.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy>=1.26,<3"], check=True)

command = [
    sys.executable,
    "-m",
    "src.models.student.train",
    "--train",
    str(train_path),
    "--dev",
    str(dev_path),
    "--out",
    str(output),
    "--epochs",
    os.environ.get("STUDENT_EPOCHS", "8"),
    "--batch-size",
    os.environ.get("STUDENT_BATCH_SIZE", "128"),
    "--max-len",
    os.environ.get("STUDENT_MAX_LEN", "512"),
    "--hidden-dim",
    os.environ.get("STUDENT_HIDDEN_DIM", "256"),
    "--num-layers",
    os.environ.get("STUDENT_NUM_LAYERS", "2"),
    "--device",
    "cuda",
]
print("Running:", " ".join(command), flush=True)
env = dict(os.environ)
env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "ru_gliner_hybrid_v4"), env.get("PYTHONPATH", "")])
subprocess.run(command, cwd=ROOT, env=env, check=True)

archive = output.parent / "student-pii.tar.gz"
with tarfile.open(archive, "w:gz") as tar:
    tar.add(output, arcname=output.name)
print(f"Download archive: {archive}", flush=True)
print(f"Student checkpoint: {output}", flush=True)