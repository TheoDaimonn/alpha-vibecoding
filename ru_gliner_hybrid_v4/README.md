# Russian GLiNER PII training bundle — hybrid v4

This bundle fine-tunes GLiNER for the Russian PII categories in the hackathon task.
The corpus policy is **ready Russian datasets first + a focused synthetic supplement
only where public supervision is missing/weak**.

Start here:

```bash
bash scripts/make_corpus.sh
cat data/hybrid/audit.json
```

Then follow [`TRAINING.md`](TRAINING.md).

Key files:

- `COVERAGE.md` — exactly which task class comes from which source;
- `scripts/fetch_redmadrobot_train.py` — task-oriented Russian PII train source;
- `scripts/build_real_corpus.py` — NEREL + FactRuEval/Kaggle ingestion;
- `ru_pii/synthetic_supplement.py` — focused task-gap supplement;
- `scripts/build_hybrid_corpus.py` — merge/dedupe/coverage guards;
- `scripts/train.py` — fine-tuning + threshold calibration + post-train benchmark;
- `ru_pii/inference.py` — Python inference wrapper;
- `notebooks/01_ru_gliner_finetune.ipynb` — notebook workflow.

The archive intentionally does **not** contain trained weights. The current execution
environment did not have usable GPU/model-download capability. Training metrics are
therefore produced only when you run the full command on your GPU machine.
