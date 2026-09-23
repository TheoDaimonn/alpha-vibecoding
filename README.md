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

## Доступ из интернета, в том числе через раздачу с телефона

```bash
docker compose up --build
```

После готовности API контейнер `tunnel` подключается к localhost.run и выводит:

```text
tunnel-1 | PUBLIC_URL=https://случайное-имя.lhr.life
tunnel-1 | DOCS_URL=https://случайное-имя.lhr.life/docs
tunnel-1 | PROCESS_URL=https://случайное-имя.lhr.life/process
```

Это фактический адрес текущего SSH-туннеля. Регистрация, карта и свой домен не нужны.
После обрыва соединения контейнер переподключается и печатает новый адрес. Старый
адрес удаляется из файла состояния, чтобы не выдавать его как актуальный.
Для фонового запуска и получения ссылки:

```bash
docker compose up -d --build
docker compose logs -f tunnel
# Только текущая ссылка, без адресов прошлых соединений:
docker compose exec -T tunnel cat /run/tunnel/public-url
```

До подключения или во время переподключения файла public-url нет. Компьютер,
Docker и интернет должны оставаться включёнными. Исходящий SSH на localhost.run:22
должен быть доступен у оператора. Адрес не закреплён на неделю; бесплатный сервис
ограничивает скорость, 2000 RPS не подтверждены нагрузочным тестом.
[Условия localhost.run](https://localhost.run/docs/forever-free/).

По умолчанию API доступен всем по ссылке. `API_KEYS` в `.env` включает проверку
`X-API-Key`. Публичный трафик проходит через localhost.run. Redis доступен только
внутри Compose; RabbitMQ публикуется только на localhost.

Если локальный порт 8000 занят уже запущенным API, используй
`API_PORT=8001 docker compose up --build`: публичный туннель всё равно подключается
к контейнеру api:8000. Отключить публикацию: `docker compose stop tunnel`.
Локальный запуск: `docker compose up -d --build redis api` (ранее работающий tunnel
нужно отдельно остановить).

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

## Сборка датасета

Корпус собирается из трёх источников и объединяется в `data/hybrid`:

1. **`data/real`** — реальные русскоязычные корпуса (NEREL, FactRuEval/Kaggle),
   собираются скриптом `build_real_corpus.py`.
2. **`data/redmadrobot`** — task-ориентированный русский PII-корпус (MIT),
   скачивается скриптом `fetch_redmadrobot_train.py`.
3. **`data/supplement`** — фокусный синтетический дополняющий корпус для классов,
   которых нет в публичных данных (`ru_pii/synthetic_supplement.py`).

Всё собирается одной командой (ставит зависимости, качает/генерирует источники,
дедуплицирует, проверяет покрытие всех классов и падает при нарушении политики):

```bash
cd ru_gliner_hybrid_v4
bash scripts/make_corpus.sh
cat data/hybrid/audit.json   # отчёт о покрытии и источниках
```

Результат — `data/hybrid/{train,dev,test}.jsonl` + `manifest.json` + `audit.json`.
Политика: сначала готовые публичные корпуса, синтетика — только закрывает пробелы
и не должна превышать 45% train. Бенчмарки в train не попадают (остаются held-out).

> **Важно:** датасеты (JSONL) не коммитятся в git — они слишком большие и
> пересоздаются скриптом. После клонирования репозитория сначала выполните
> `make_corpus.sh`.

## Дообучение GLiNER (teacher)

Дообучение teacher-модели (deberta-v3-small) на собранном корпусе:

```bash
cd ru_gliner_hybrid_v4
# Smoke-прогон (2 шага, проверка интеграции, без GPU-требований):
python scripts/train.py --config configs/train_small.json --smoke

# Полный прогон на GPU (2000 шагов + калибровка порогов на dev + бенчмарк):
python scripts/train.py --config configs/train_small.json --device cuda
```

Ключевые флаги `train.py`:

| Флаг | Назначение |
|---|---|
| `--config` | JSON-конфиг (пути, шаги, LR, батч). По умолчанию `configs/train_small.json` |
| `--model` | Локальная директория GLiNER или явно разрешённый Hub-модель |
| `--device` | `auto` \| `cpu` \| `cuda` |
| `--smoke` | 2 шага / 48 train / 12 dev — только проверка интеграции |
| `--external` | Доп. канонический train-JSONL (суммарно ≤ 25% от base train) |
| `--allow-download` | Разрешить скачивание модели с Hub (по умолчанию offline) |
| `--skip-benchmark` | Пропустить калибровку порогов и пост-тренировочный бенчмарк |

Результат — директория `artifacts/gliner-ru-pii-small/` с весами
(`model.safetensors`), `labels.json`, `thresholds.json`, `training_log.jsonl` и
`benchmark.json`. Полный прогон на CPU требует явного `--allow-slow-cpu`.

## Дистилляция студента (teacher → student)

Дистилляция поднимает recall студента до уровня teacher: teacher прогоняется по
корпусу, его предсказания сливаются с золотыми метками в расширенный корпус
(`train_expanded.jsonl`), и на нём обучается студент.

### Локально (CPU/MPS)

```bash
python -m src.models.student.train \
  --train ru_gliner_hybrid_v4/data/hybrid/train.jsonl \
  --dev ru_gliner_hybrid_v4/data/hybrid/dev.jsonl \
  --teacher ru_gliner_hybrid_v4/models/gliner-ru-pii-small \
  --out artifacts/student-pii.pt \
  --epochs 8 --device mps
```

Флаг `--teacher <gliner-model-dir>` добавляет soft-метки teacher на train-тексты.
Без него студент учится только на золотых метках.

### На GPU (Kaggle, 2×T4)

Тяжёлый teacher (644MB) и большой корпус гоняются на Kaggle. Скрипт
`scripts/push_kaggle_distill.sh` собирает бандл (teacher + корпус + `src`),
заливает его как приватный датасет и запускает kernel `kaggle_distill_train.py`:

```bash
# Требуется kaggle CLI и ключ (KAGGLE_KEY или ~/.kaggle/kaggle.json)
bash scripts/push_kaggle_distill.sh
```

Пайплайн `kaggle_distill_train.py`:
1. Устанавливает зависимости из `requirements-distill.txt`.
2. Прогоняет teacher по train-корпусу на GPU → `train_expanded.jsonl`.
3. Обучает студента на расширенном корпусе (8 эпох, cuda).
4. Упаковывает чекпоинт в `student-pii.tar.gz` для скачивания.

Параметры студента на Kaggle задаются env: `STUDENT_EPOCHS`, `STUDENT_BATCH_SIZE`,
`STUDENT_MAX_LEN`, `STUDENT_HIDDEN_DIM`, `STUDENT_NUM_LAYERS`.

## Безопасность

- Исходные ПД не попадают в логи (логируются только типы и метрики).
- Allowlist систем через `API_KEYS` (`X-API-Key`).
- Маскирование сохраняет позиции символов (важно для LLM-контекста).
- Демаскирование доступно только по `payload_id` (корреляция).
## ONNX RuBERT: настройки запуска

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
`THRESHOLD` относится к GLiNER, не к ONNX BIO-декодеру.

`API_PORT` меняет порт хоста; внутри Compose API всегда слушает 8000.
Для Python напрямую доступны `HOST` и `PORT`. `API_KEYS` задаёт ключи через запятую.
Меняя путь модели в контейнере, обеспечьте наличие каталога внутри образа или
подключите volume. Изменения `.env` применяются пересозданием контейнера.
