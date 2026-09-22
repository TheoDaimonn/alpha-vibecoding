# Модуль безопасности персональных данных (PII Security Module)

Сервис-прокси для защиты персональных данных при обращении к LLM. Реализует
контракт `POST /process`: идентификация ПД → маскирование → демаскирование.

## Быстрый старт

```bash
# 1. Зависимости (Python 3.10+)
pip install -r requirements.txt

# 2. Redis (для correlation store)
brew install redis && brew services start redis

# 3. Запуск API (4 воркера, rules-детектор)
DETECTOR=rules CORRELATION_STORE=redis API_WORKERS=4 python -m src.run_api

# 4. Проверка
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{"payload":"Клиент Иванов Иван Иванович, email: test@example.org","payload_id":"demo-1"}'
```

Или через Docker:

```bash
docker compose up --build
```

## Контракт API

`POST /process` — единый эндпоинт для маскирования и демаскирования.

```json
// Запрос
{ "payload": "<строка>", "payload_id": "<идентификатор>" }
// Ответ 200
{ "result": "<строка>" }
```

Логика (по ТЗ):
- Первый запрос с новым `payload_id` → **маскирование** → возвращает маску и
  запоминает соответствие по `payload_id`.
- Второй запрос с тем же `payload_id` (payload = ваша маска) → **демаскирование**
  → возвращает исходную строку.
- Эндпоинт идемпотентен по `payload_id` (безопасен к ретраям).
- При перегрузке возвращает `429` с `Retry-After`.

## Архитектура

```
Система-потребитель → POST /process (FastAPI, async)
                          │
                          ├─ Correlation store (Redis) — payload_id → {original, masked, spans}
                          │
                          └─ InferenceEngine (in-process, по умолчанию)
                               └─ пул воркер-потоков с батчингом
                                    └─ Detector (rules | gliner | student | hybrid)
```

- **FastAPI (async)** держит тысячи конкурентных соединений без блокировки.
- **InferenceEngine** — in-memory очередь + пул потоков, которые батчат запросы
  и гоняют их через детектор. Результаты возвращаются через asyncio futures.
- **Redis** хранит соответствие `payload_id` → маска/оригинал для демаскирования
  и идемпотентности. Общий для всех воркеров.
- **RabbitMQ** опционален (`TRANSPORT=rabbitmq`) для распределённого деплоя.

## Настройка (env)

| Переменная | По умолчанию | Описание |
|---|---|---|
| `DETECTOR` | `gliner` | Движок: `gliner` \| `student` \| `rules` \| `hybrid` |
| `MODEL_PATH` | `ru_gliner_hybrid_v4/models/gliner-ru-pii-small` | Путь к GLiNER |
| `STUDENT_MODEL_PATH` | `artifacts/student-pii.pt` | Путь к дистиллированной модели |
| `CORRELATION_STORE` | `memory` | `memory` (быстро, 1 воркер) \| `redis` (общий) |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis |
| `TRANSPORT` | `inprocess` | `inprocess` (быстро) \| `rabbitmq` (распределённо) |
| `RABBITMQ_URL` | `amqp://guest:guest@localhost:5672/` | RabbitMQ |
| `API_WORKERS` | `1` | Число uvicorn-воркеров (каждый со своим движком) |
| `WORKER_THREADS` | `4` | Потоков на движок |
| `WORKER_BATCH_SIZE` | `32` | Размер батча |
| `WORKER_BATCH_TIMEOUT_S` | `0.02` | Таймаут накопления батча |
| `API_KEYS` | (пусто) | Allowlist ключей через `X-API-Key` (через запятую) |

## Детекторы

- **`rules`** — быстрый regex-матчер для паттерн-типов (email, телефон, карта,
  CVV, PIN, ИНН, даты, паспорт, индекс, IP, URL, СНИЛС, ОМС). ~0 мс на запрос.
- **`gliner`** — полная GLiNER-модель (deberta-v3-small). Высокое качество,
  но ~16 текстов/с на CPU.
- **`student`** — дистиллированная BiLSTM-CRF (char-level). ~650 текстов/с на
  CPU. Тренируется из `src/models/student/train.py`.
- **`hybrid`** — rules для паттерн-типов + модель для семантических (ФИО, адрес).

## Производительность

На 4-ядерном Mac (CPU), `API_WORKERS=8`, `WORKER_THREADS=2`, `CORRELATION_STORE=redis`:

| Детектор | RPS (1000 конкурентных) | p50 latency | p99 latency |
|---|---|---|---|
| `rules` | ~1290 | ~570 мс | ~720 мс |
| `student` | ~2490 | ~330 мс | ~380 мс |

Целевые показатели ТЗ (RPS 1000, latency ≤ 1 с) достигаются. **`student`** —
рекомендуемый детектор: покрывает все типы ПД (включая семантические ФИО/адрес)
и быстрее `rules`. Для максимальной скорости используйте `rules` (только
паттерн-типы).

Ключевой инсайт для масштабирования: для быстрой модели (student/rules)
используйте **больше воркеров с меньшим числом потоков** (8 воркеров × 2 потока),
чтобы избежать конкуренции за тензор внутри воркера.

## Тесты

```bash
# Функциональные (без модели/Redis/RabbitMQ)
pytest tests/test_process.py

# Нагрузочный тест: 1000 конкурентных запросов, замер latency
python -m tests.test_load
# Настройки: LOAD_CONCURRENCY, LOAD_COUNT, LOAD_URL
```

## Дистилляция студента

```bash
python -m src.models.student.train \
  --train ru_gliner_hybrid_v4/data/hybrid/train.jsonl \
  --dev ru_gliner_hybrid_v4/data/hybrid/dev.jsonl \
  --out artifacts/student-pii.pt \
  --epochs 3 --device cpu
```

Студент — char-level BiLSTM-CRF, обучается на корпусе (char-спаны). Быстрее
GLiNER в ~40 раз на CPU.

## Безопасность

- Исходные ПД не попадают в логи (логируются только типы и метрики).
- Allowlist систем через `API_KEYS` (`X-API-Key`).
- Маскирование сохраняет позиции символов (важно для LLM-контекста).
- Демаскирование доступно только по `payload_id` (корреляция).