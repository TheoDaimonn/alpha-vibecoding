# v4 corpus/training report

## Goal

Prepare GLiNER fine-tuning data for every PII type required by the Russian banking
hackathon while avoiding a synthetic-only corpus.

## Corpus policy

- Primary task-oriented source: `redmadrobot-rnd/pii_train` (MIT), 17,137 published
  Russian training rows / 39,687 entity spans. Its dataset card documents a mix of
  pseudonymized production-log contexts, synthetic document rows and hard negatives.
- Human-written Russian text: NEREL and FactRuEval-2016.
- Focused local supplement: 12,000 train + 1,500 dev + 1,500 test rows, used for task
  labels not directly available in the public sources (passport issuer/division/date,
  citizenship, postal/apartment, CVV/PIN/cardholder, fine-grained document parts, etc.).
- External test corpora remain held out: `hivetrace/pii-bench` and
  `redmadrobot-rnd/pii_benchmark`.

## Required-label coverage

`ru_pii.schema.TASK_REQUIRED_LABELS` contains every field required by the task.
`build_hybrid_corpus.py` refuses to create the final corpus when any required label
has fewer than 250 train spans by default, when no sourced rows are present, or when
the local pure-synthetic supplement would exceed 45% of train rows.

The packaged supplement audit has no missing required labels. The final hybrid counts
are produced on the user's machine after the published source corpora are downloaded.

## Verification performed in this delivery

- deterministic supplement generation completed;
- canonical offsets and labels validated;
- all required labels appear in train/dev/test supplement splits;
- parser and corpus unit tests pass;
- hybrid merge/validation was smoke-tested with a small sourced fixture;
- actual GLiNER fine-tuning and real benchmark inference were **not** executed in this
  environment, so no model-quality number is claimed.

See `TRAINING.md` for exact commands and `COVERAGE.md` for source-by-class details.
