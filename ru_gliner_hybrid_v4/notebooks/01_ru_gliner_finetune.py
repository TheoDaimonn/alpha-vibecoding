# %% [markdown]
# # Fine-tune GLiNER on the Russian PII hybrid corpus
#
# Ready Russian corpora are the base; a focused synthetic supplement closes only
# task-specific gaps. See `COVERAGE.md` and `TRAINING.md`.

# %%
from pathlib import Path
import json, subprocess, sys
ROOT = Path.cwd()
if not (ROOT / 'scripts').exists():
    ROOT = ROOT.parent
print('root:', ROOT)

# %% [markdown]
# ## 1. Build sourced + supplement + hybrid corpus
# Enable on a machine with internet access (Kaggle/Hugging Face downloads).

# %%
RUN_BUILD = False
if RUN_BUILD:
    subprocess.run(['bash', 'scripts/make_corpus.sh'], cwd=ROOT, check=True)
else:
    print('Skipped. Run: bash scripts/make_corpus.sh')

# %%
for path in ['data/supplement/audit.json', 'data/hybrid/audit.json']:
    p = ROOT / path
    if p.exists():
        print('\n###', path)
        print(json.dumps(json.loads(p.read_text(encoding='utf-8')), ensure_ascii=False, indent=2)[:16000])
    else:
        print(path, 'not present yet')

# %% [markdown]
# ## 2. Validate all hackathon classes

# %%
RUN_VALIDATE = False
if RUN_VALIDATE:
    subprocess.run([
        sys.executable, 'scripts/validate_data.py', '--data-dir', 'data/hybrid',
        '--mode', 'hybrid', '--min-train-support', '250'
    ], cwd=ROOT, check=True)

# %% [markdown]
# ## 3. Download the pinned base model

# %%
RUN_MODEL_DOWNLOAD = False
if RUN_MODEL_DOWNLOAD:
    subprocess.run([
        sys.executable, 'scripts/download_model.py',
        '--model', 'gliner-community/gliner_small-v2.5',
        '--revision', 'f227d3cd637bd4e6757ae143935316d062393341',
        '--output', 'models/base-small',
    ], cwd=ROOT, check=True)

# %% [markdown]
# ## 4. Preflight / smoke

# %%
RUN_PREFLIGHT = False
if RUN_PREFLIGHT:
    subprocess.run([
        sys.executable, 'scripts/train.py', '--config', 'configs/train_small.json',
        '--model', 'models/base-small', '--device', 'cuda', '--prepare-only',
        '--output', 'artifacts/preflight-small'
    ], cwd=ROOT, check=True)

# %%
RUN_SMOKE = False
if RUN_SMOKE:
    subprocess.run([
        sys.executable, 'scripts/train.py', '--config', 'configs/train_small.json',
        '--model', 'models/base-small', '--device', 'cuda', '--smoke',
        '--output', 'artifacts/smoke-small'
    ], cwd=ROOT, check=True)

# %% [markdown]
# ## 5. Full fine-tuning + automatic Russian benchmarks

# %%
RUN_FULL_TRAIN = False
if RUN_FULL_TRAIN:
    subprocess.run([
        sys.executable, 'scripts/train.py', '--config', 'configs/train_small.json',
        '--model', 'models/base-small', '--device', 'cuda',
        '--output', 'artifacts/gliner-ru-pii-small', '--benchmark-external', 'required'
    ], cwd=ROOT, check=True)

# %% [markdown]
# ## 6. Re-run benchmarks without retraining

# %%
RUN_BENCHMARK = False
if RUN_BENCHMARK:
    subprocess.run([
        sys.executable, 'scripts/post_train_benchmark.py',
        '--model', 'artifacts/gliner-ru-pii-small', '--device', 'cuda',
        '--external', 'required', '--output', 'reports/rebenchmark.json'
    ], cwd=ROOT, check=True)

# %% [markdown]
# ## 7. Python inference wrapper

# %%
RUN_INFERENCE = False
if RUN_INFERENCE:
    from ru_pii.inference import RussianPIIDetector
    detector = RussianPIIDetector.from_pretrained('artifacts/gliner-ru-pii-small', device='cuda', batch_size=8)
    text = 'Клиент Иванов Иван Иванович, паспорт серия 4509 номер 123456, email test@example.org.'
    print([e.to_dict() for e in detector.predict(text)])
