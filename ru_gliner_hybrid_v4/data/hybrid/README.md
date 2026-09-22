# data/hybrid

This directory is materialized by `bash scripts/make_corpus.sh` on a machine with
Kaggle/Hugging Face access. It is intentionally not prefilled in the archive because
third-party source corpora are not redistributed here.

The final files are `train.jsonl`, `dev.jsonl`, `test.jsonl`, `audit.json`, and
`manifest.json`. The builder refuses synthetic-only training and verifies all required
hackathon PII labels before writing a successful corpus.
