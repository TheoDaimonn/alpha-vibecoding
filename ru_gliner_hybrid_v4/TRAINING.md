# Training GLiNER on the Russian PII hybrid corpus

## 0. What this version trains on

The default corpus combines ready Russian datasets with a focused synthetic
supplement. Every PII class required by the hackathon has supervised train spans;
see `COVERAGE.md`. External benchmarks (`redmadrobot-rnd/pii_benchmark` and
`hivetrace/pii-bench`) never enter train/dev.

## 1. Environment

Create an environment and install data tooling:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
python -m pip install -U pip
python -m pip install -r requirements-data.txt
```

For training, install a PyTorch build appropriate for your CUDA version first,
then:

```bash
python -m pip install -r requirements-training.txt
```

Check CUDA:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

## 2. Build the corpus

One command:

```bash
bash scripts/make_corpus.sh
```

It performs, in order:

1. NEREL + FactRuEval (Kaggle first, HF fallback) -> `data/real/`;
2. `redmadrobot-rnd/pii_train` -> `data/redmadrobot/train.jsonl`;
3. deterministic focused supplement -> `data/supplement/`;
4. merge/dedup/coverage checks -> `data/hybrid/`;
5. integrity + class-support validation.

The shipped `data/supplement/` is already generated for convenience, but the script
regenerates it deterministically.

Inspect the final audit:

```bash
cat data/hybrid/audit.json
cat data/hybrid/manifest.json
```

`task_required_support` / `task_required_train_support` must contain every field
listed in `COVERAGE.md` and no required class may be below the configured minimum.

### Build pieces manually

```bash
python scripts/build_real_corpus.py \
  --output data/real \
  --factru-backend auto \
  --skip-collection3

python scripts/fetch_redmadrobot_train.py \
  --output data/redmadrobot

python -m ru_pii.synthetic_supplement \
  --output data/supplement

python scripts/build_hybrid_corpus.py \
  --real-dir data/real \
  --redmadrobot-dir data/redmadrobot \
  --supplement-dir data/supplement \
  --output data/hybrid \
  --min-train-support 250

python scripts/validate_data.py \
  --data-dir data/hybrid \
  --mode hybrid \
  --min-train-support 250
```

## 3. Download the base model

Pin the same base revision used by the config:

```bash
python scripts/download_model.py \
  --model gliner-community/gliner_small-v2.5 \
  --revision f227d3cd637bd4e6757ae143935316d062393341 \
  --output models/base-small
```

## 4. Preflight

Checks real tokenizer alignment/windowing without doing optimiser steps:

```bash
python scripts/train.py \
  --config configs/train_small.json \
  --model models/base-small \
  --device cuda \
  --prepare-only \
  --output artifacts/preflight-small
```

## 5. Smoke train

Two optimisation steps only; integration check, not a quality measurement:

```bash
python scripts/train.py \
  --config configs/train_small.json \
  --model models/base-small \
  --device cuda \
  --smoke \
  --output artifacts/smoke-small
```

## 6. Full train + automatic metrics

```bash
python scripts/train.py \
  --config configs/train_small.json \
  --model models/base-small \
  --device cuda \
  --output artifacts/gliner-ru-pii-small \
  --benchmark-external required
```

At the end the script:

1. selects the best checkpoint by dev loss;
2. calibrates per-label thresholds on **dev only**;
3. evaluates `data/hybrid/test.jsonl`;
4. separately evaluates the real-only held-out set when present;
5. separately evaluates the synthetic all-task coverage test;
6. runs Russian `hivetrace/pii-bench` and `redmadrobot-rnd/pii_benchmark`;
7. writes `benchmark.json`, `benchmark.md`, `thresholds.json`, `RUN_STATUS.json`.

For an offline run use `--benchmark-external none`. `auto` tries external benchmarks
and records `skipped` if the network is unavailable; `required` fails the run if a
benchmark cannot be loaded.

## 7. Re-run benchmarks without retraining

```bash
python scripts/post_train_benchmark.py \
  --model artifacts/gliner-ru-pii-small \
  --device cuda \
  --external required \
  --output reports/rebenchmark.json
```

## 8. Python inference

```python
from ru_pii.inference import RussianPIIDetector

detector = RussianPIIDetector.from_pretrained(
    "artifacts/gliner-ru-pii-small",
    device="cuda",
    batch_size=8,
)

entities = detector.predict(
    "Клиент Иванов Иван Иванович, email: test@example.org",
    include_auxiliary=False,
)
for entity in entities:
    print(entity.to_dict())
```

For long documents the wrapper uses overlapping windows and returns offsets in the
original Python string. Do not infer API/masking policy directly from a NER label;
that layer will be designed separately.
