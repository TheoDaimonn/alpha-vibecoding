# data/real

Здесь после запуска `scripts/build_real_corpus.py` появятся:

- `train.jsonl`
- `dev.jsonl`
- `test.jsonl`
- `manifest.json`
- `audit.json`

В delivery ZIP сами сторонние корпуса намеренно не перепакованы. Сборка выполняется из upstream готовых наборов:

```bash
python scripts/build_real_corpus.py --output data/real --factru-backend auto
python scripts/validate_data.py --data-dir data/real
```

`auto` сначала пытается получить FactRuEval через Kaggle `constantinwerner/multilingual-ner-dataset`, а затем использует HF mirror. NEREL и Collection3 загружаются из их публичных репозиториев.

`audit.json` должен содержать `"synthetic_rows": 0`.
