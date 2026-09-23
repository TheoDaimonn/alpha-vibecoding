# Coverage of the hackathon PII classes

The default v4 corpus is **hybrid**. It does not pretend that all sensitive Russian
banking fields exist in public real-world corpora. Public/pseudonymized annotated
sources are used wherever they have a trustworthy label; a focused synthetic
supplement closes the remaining task gaps.

## Default source mix

1. **redmadrobot-rnd/pii_train (MIT)** — 17,137 Russian rows / 39,687 published spans.
   The dataset card says it combines 9,940 pseudonymized production-log contexts,
   6,437 synthetic document-style rows and 760 hard negatives. Row-level provenance
   is not published, so this project keeps the source as a documented `mixed` source.
   Its paired `pii_benchmark` is never used for training.
2. **NEREL** — human-written Russian Wikinews, manually annotated nested NER and
   relations. `DATE_OF_BIRTH` / `PLACE_OF_BIRTH` relations are the only generic
   source that becomes `BIRTH_DATE` / `BIRTH_PLACE`.
3. **FactRuEval-2016** — human-written Russian news, manual NER; the build tries the
   Kaggle multilingual mirror first. Only `PERSON` supervision is imported because
   a generic news LOC is not the same thing as a private address.
4. **Focused task supplement** — 12,000 train + 1,500 dev + 1,500 test rows generated
   locally with non-issued values and exact offsets. It exists specifically for
   task classes absent or weak in public Russian corpora.

Collection3 is available as an optional source but is disabled by default because
its Hugging Face licence metadata is `other` and should be reviewed before use.

## Required task coverage

| Task field | Canonical label(s) | Sourced supervision | Focused supplement |
|---|---|---|---|
| ФИО | `PERSON` | RedMadRobot, NEREL, FactRuEval | yes |
| Дата рождения | `BIRTH_DATE` | NEREL relation | yes |
| Место рождения | `BIRTH_PLACE` | NEREL relation | yes |
| Серия/номер паспорта | `PASSPORT`, `PASSPORT_SERIES`, `PASSPORT_NUMBER` | RedMadRobot has coarse `PASSPORT` | **yes, fine-grained** |
| Гражданство | `CITIZENSHIP` | no safe direct mapping by default | **yes** |
| Орган выдачи паспорта | `PASSPORT_ISSUER` | no safe direct mapping by default | **yes** |
| Код подразделения | `PASSPORT_DIVISION_CODE` | no | **yes** |
| Дата выдачи паспорта | `PASSPORT_ISSUE_DATE` | no | **yes** |
| Серия/номер ВУ | `DRIVER_LICENSE`, `DRIVER_LICENSE_SERIES`, `DRIVER_LICENSE_NUMBER` | RedMadRobot has coarse `DRIVER_LICENSE` | **yes, fine-grained** |
| Адрес | `ADDRESS` | components in RedMadRobot | **yes, complete short span** |
| Страна | `COUNTRY` | RedMadRobot, NEREL | yes |
| Индекс | `POSTAL_CODE` | no default source | **yes** |
| Город | `CITY` | RedMadRobot, NEREL | yes |
| Улица | `STREET` | RedMadRobot | yes |
| Дом | `HOUSE` | RedMadRobot | yes |
| Квартира | `APARTMENT` | no default source | **yes** |
| Email | `EMAIL` | RedMadRobot | yes |
| Телефон | `PHONE` | RedMadRobot | yes |
| ИНН | `INN` | RedMadRobot | yes |
| Номер карты | `CARD_NUMBER` | RedMadRobot | yes |
| CVV | `CVV` | no public production-data source expected | **yes** |
| PIN | `PIN` | no public production-data source expected | **yes** |
| Имя держателя | `CARDHOLDER` | no direct source label | **yes** |

Additional address labels: `REGION`, `DISTRICT`, `BUILDING`.
Additional bonus PII labels imported from RedMadRobot: `SNILS`, `OMS`, `MILITARY_ID`,
`BIRTH_CERTIFICATE`, `IP_ADDRESS`, `URL`, plus generic `OTHER_ID` support.

## Hard guardrails in the builder

`build_hybrid_corpus.py` fails when:

- `data/redmadrobot/train.jsonl` is missing;
- no sourced/mixed training rows are present;
- any required hackathon class has fewer than the configured minimum number of train spans;
- the local pure-synthetic supplement exceeds 45% of train rows by default;
- exact normalized texts leak from train into dev/test.

The 45% check concerns the **local supplement only**. RedMadRobot itself is a published
mixed corpus and does not expose row-level source-family provenance, so its internal
synthetic share cannot honestly be separated by this project.
