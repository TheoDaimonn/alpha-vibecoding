# PII API — runtime

Ветка содержит student и RuBERT-tiny2 ONNX int8, инференс, FastAPI, Redis и SSH-туннель.
Обучения, корпусов, ноутбуков и GLiNER-весов здесь нет.

```bash
git clone --depth 1 --single-branch --branch runtime-only https://github.com/TheoDaimonn/alpha-vibecoding.git
cd alpha-vibecoding
git lfs install
git lfs pull --include="artifacts/**"
docker compose up --build
```

API: http://localhost:8000/docs. После подключения туннель печатает PUBLIC_URL.
Текущая ссылка: `docker compose exec -T tunnel cat /run/tunnel/public-url`.
Для локального запуска: `docker compose up --build redis api`.

`POST /process`: `{"payload":"текст", "payload_id":"уникальный-id"}` → `{"result":"маска"}`.
Повторите запрос с маской и тем же ID для демаскирования.
`/health`, `/metrics`, `/ui` — состояние, метрики и ручная проверка.

Переменные Compose: `API_PORT=8000`, `API_WORKERS=1`, `API_KEYS` (список через
запятую, клиент передаёт X-API-Key). По умолчанию API доступен без ключа.
Redis используется для общего хранения корреляций между процессами.

Student обрабатывает до 512 символов. Recall 0.95 и 2000 RPS не подтверждены.
Бесплатный туннель localhost.run имеет ограничения, адрес может меняться.
RuBERT-base из Kaggle ещё не интегрирован; ONNX здесь — отдельный rubert-tiny2.

Для запуска без Docker: Python 3.10+, `pip install -r requirements.txt`,
затем `python -m src.run_api` из корня. Локально по умолчанию один процесс
и хранилище в памяти; для нескольких процессов необходим общий Redis.



`cp .env.example .env && docker compose up --build` запускает ONNX-детектор.
Без `.env` остаётся student. Переменные экспортированного окружения имеют
приоритет над `.env`. Для прямого запуска Python переменные надо экспортировать;
пути `/app/...` из примера предназначены для контейнера.

`RUBERT_MODEL_PATH` — каталог ONNX и tokenizer; `RUBERT_MAX_LEN` / `RUBERT_STRIDE`
задают окно и перекрытие (пустые значения берутся из checkpoint).
`MODEL_BATCH_SIZE` — число окон в ONNX-батче, `ONNX_INTRA_THREADS` /
`ONNX_INTER_THREADS` — потоки ONNX Runtime; провайдер CPU.
Невалидное окно, stride, batch или число потоков приводят к ошибке при загрузке.
`WORKER_*` управляют очередью API; `API_WORKERS` — число процессов с отдельной
копией модели. По умолчанию один процесс. `POSTPROCESS` — дополнительный
эвристический фильтр, выключен; PUBLIC-метки модели исключаются независимо от него.
`THRESHOLD` не применяется к ONNX BIO-декодеру.

`API_PORT` меняет порт хоста; внутри Compose API всегда слушает 8000.
Для Python напрямую доступны `HOST` и `PORT`. `API_KEYS` задаёт ключи через запятую.
Меняя путь модели в контейнере, обеспечьте наличие каталога внутри образа или
подключите volume. Изменения `.env` применяются пересозданием контейнера.
