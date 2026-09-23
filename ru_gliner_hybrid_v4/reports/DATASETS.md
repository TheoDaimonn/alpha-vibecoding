# Dataset inventory

See `COVERAGE.md` for the class-by-class matrix.

## Training

- `redmadrobot-rnd/pii_train` — primary task-oriented Russian PII corpus, MIT,
  17,137 published rows / 39,687 spans; documented mixture of pseudonymized production
  contexts, synthetic document rows, and hard negatives.
- NEREL — human-written Russian Wikinews, manual nested NER/relations. Used for PERSON,
  geographical components, and relation-derived birth date/place.
- FactRuEval-2016 — human-written Russian news, manual NER. Used conservatively for PERSON.
- Local focused task supplement — 12k/1.5k/1.5k train/dev/test. Fills classes not directly
  available in the above sources and provides exact offsets + hard negatives.

## Optional

- Collection3 — useful PERSON data, disabled by default pending licence review.
- Jay Guard benchmark — optional de-identified conversational context; disabled by default
  because consuming a benchmark for training removes its value as an independent test.

## Evaluation only

- `hivetrace/pii-bench`
- `redmadrobot-rnd/pii_benchmark`
