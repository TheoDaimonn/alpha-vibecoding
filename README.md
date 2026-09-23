# PII API — runtime

Ветка содержит student-модель, инференс, FastAPI, Redis и публичный SSH-туннель.
Обучения, корпусов, ноутбуков и GLiNER/transformer-весов здесь нет.

```bash
git clone --single-branch --branch runtime-only https://github.com/TheoDaimonn/alpha-vibecoding.git
cd alpha-vibecoding
git lfs install
git lfs pull --include="artifacts/student-pii.pt"
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
Новый RuBERT из Kaggle в эту поставку ещё не интегрирован.

Для запуска без Docker: Python 3.10+, `pip install -r requirements.txt`,
затем `python -m src.run_api` из корня. Локально по умолчанию один процесс
и хранилище в памяти; для нескольких процессов необходим общий Redis.
