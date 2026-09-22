"""Kaggle entrypoint: distill the GLiNER teacher into the student.

Pipeline:
  1. Run the GLiNER teacher over the corpus on GPU to get its predictions.
  2. Merge teacher predictions with the gold labels into an expanded corpus.
  3. Train the student (char-level BiLSTM-CRF) on the expanded corpus.

Training on teacher predictions (in addition to gold labels) raises the
student's recall toward the teacher's level, which is the goal (accuracy >= 0.95).
"""
from __future__ import annotations

import json
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

output = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working/artifacts/student-pii.pt"))
output.parent.mkdir(parents=True, exist_ok=True)

candidates = sorted(input_root.rglob("*/ru_gliner_hybrid_v4/data/hybrid/train.jsonl"))
if not candidates:
    raise SystemExit("No training corpus found")
train_path = candidates[0]
dev_path = train_path.parent / "dev.jsonl"

# --- Install deps (gliner teacher + torch) before importing ---
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements-distill.txt")], check=True)

# --- 1. Generate teacher predictions on the training corpus ---
from gliner import GLiNER  # noqa: E402
from ru_pii.inference import RussianPIIDetector  # noqa: E402

teacher = RussianPIIDetector.from_pretrained(
    str(ROOT / "ru_gliner_hybrid_v4" / "models" / "gliner-ru-pii-small"),
    device="cuda",
    batch_size=32,
)
print("Teacher loaded", flush=True)

expanded = Path("/kaggle/working") / "train_expanded.jsonl"
with train_path.open(encoding="utf-8") as fin, expanded.open("w", encoding="utf-8") as fout:
    texts = []
    rows = []
    for line in fin:
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(row)
        texts.append(row["text"])
        if len(texts) >= 64:
            preds = teacher.predict_batch(texts)
            for r, ents in zip(rows, preds):
                gold = {(e["start"], e["end"], e["label"]) for e in r.get("entities", [])}
                merged = list(gold)
                for e in ents:
                    key = (e.start, e.end, e.label)
                    if key not in gold:
                        merged.append({"start": e.start, "end": e.end, "label": e.label, "text": e.text, "privacy": "personal"})
                r["entities"] = merged
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            texts = []
            rows = []
    if texts:
        preds = teacher.predict_batch(texts)
        for r, ents in zip(rows, preds):
            gold = {(e["start"], e["end"], e["label"]) for e in r.get("entities", [])}
            merged = list(gold)
            for e in ents:
                key = (e.start, e.end, e.label)
                if key not in gold:
                    merged.append({"start": e.start, "end": e.end, "label": e.label, "text": e.text, "privacy": "personal"})
            r["entities"] = merged
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Expanded corpus written: {expanded}", flush=True)

# --- 2. Train the student on the expanded corpus ---
env = dict(os.environ)
env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "ru_gliner_hybrid_v4"), env.get("PYTHONPATH", "")])
command = [
    sys.executable,
    "-m",
    "src.models.student.train",
    "--train",
    str(expanded),
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
subprocess.run(command, cwd=ROOT, env=env, check=True)

archive = output.parent / "student-pii.tar.gz"
with tarfile.open(archive, "w:gz") as tar:
    tar.add(output, arcname=output.name)
print(f"Download archive: {archive}", flush=True)