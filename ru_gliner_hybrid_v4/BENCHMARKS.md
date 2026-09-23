# Benchmarks

After full training, `scripts/train.py` automatically calibrates thresholds on
`data/hybrid/dev.jsonl` and then evaluates independent sets.

Local reports:

- `data/hybrid/test.jsonl` — combined held-out test, including every required task label;
- `data/real/test.jsonl` — real-text held-out subset when built;
- `data/supplement/test.jsonl` — explicitly synthetic all-task coverage test.

External Russian held-out PII benchmarks:

- `hivetrace/pii-bench`;
- `redmadrobot-rnd/pii_benchmark` (paired held-out evaluation set for `pii_train`).

The external benchmark sets are never read by the corpus builder or training data
preparation. The scorer reports exact and overlap/coarse projections where annotation
taxonomies differ. Do not compare metrics produced under different boundary or label
projection protocols as if they were identical.
