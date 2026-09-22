# Third-party provenance and licensing notes

The archive redistributes only this project's locally generated CC0 supplement. It
does **not** bundle upstream dataset files or model weights; source corpora are fetched
on the training machine and immutable revisions are recorded in manifests.

## Default training sources

- **RedMadRobot `pii_train`** — https://huggingface.co/datasets/redmadrobot-rnd/pii_train — Hugging Face card declares **MIT**. Published as 17,137 Russian train rows with 21 PII entity types; source-family composition is documented by the publisher. Its paired `pii_benchmark` is held out.
- **NEREL** — https://huggingface.co/datasets/iluvvatar/NEREL — human-written Russian Wikinews with manual nested NER and relations. Preserve corpus/source attribution and review terms for redistribution/deployment.
- **FactRuEval-2016** — original competition corpus; default build tries Kaggle `constantinwerner/multilingual-ner-dataset` first, then an HF mirror. Preserve original attribution and review the selected mirror's terms.

## Disabled by default

- **Collection3** — HF metadata reports `license=other`; enable only after review.
- **Jay Guard NER Benchmark** — optional; published as a benchmark and therefore not consumed by default training.

## Evaluation only

- `redmadrobot-rnd/pii_benchmark`
- `hivetrace/pii-bench`

The repository license applies to this project's code and locally generated supplement only; it does not grant rights over third-party corpora.
