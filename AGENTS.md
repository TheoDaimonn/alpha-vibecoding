# Контекст проекта и проверенные факты

Документ для агентов, работающих в этом репозитории. Актуализирован по коду и
результатам сессии 2026-09-23. Статусы внешних сервисов ниже — исторические наблюдения,
а не мониторинг: перед действиями проверяйте их заново.

## Цель и требования

- Проект: прокси защиты персональных данных на FastAPI, идентификация → маскирование
  → демаскирование. Основной язык данных — русский.
- Источник требований: PDF в `rules/`. Читайте их через `pdftotext`; требования
  организаторов приоритетнее предположений в README и комментариях.
- ТЗ включает 17 групп ПД, вариации регистра/дат/документов, контекст публичности
  (поэт Пушкин, адрес отделения банка), тексты до 100 000 токенов,
  1000 RPS и latency ≤1 с. Есть дополнительные баллы за 2000 RPS.
- Пользователь отдельно требует recall ≥0.95. «Качество 95%» в ТЗ не задаёт
  однозначно exact-span протокол; не подменяйте одну метрику другой.
- Текущая задача публичного доступа: адрес примерно на неделю, короткие пики
  2000 RPS, остальное время простой. Это НЕ непрерывные 2000 RPS всю неделю
  и НЕ просто 2000 открытых соединений.
- Пользователь раздаёт интернет с телефона, VPS нет. Варианты с обязательной
  банковской картой ему не подходят; Pinggy Pro отвергнут именно по этой причине.

## Работа с репозиторием

- Не сбрасывайте незакоммиченные изменения: до текущих работ уже были изменения
  в config/factory, distillation и новые transformer/postprocess модули.
- Сначала смотрите `git status` и локальный код. Не выдавайте старые комментарии,
  рекламные цифры и предположения за измеренные свойства.
- Локальный Python: `.venv/bin/python`, Kaggle CLI: `.venv/bin/kaggle`.
- Секреты не включать в код, логи, ответы, Docker build context и Kaggle bundles.
  Не просите отправлять токены в чат; используйте штатные локальные настройки.
- Пользователь предпочитает действие без повторных вопросов, когда разрешение уже
  дано. Разрешения из сессии сохраняются, но не обходите отказ автоматической проверки.
- Не останавливайте посторонние процессы и не перезапускайте весь Docker Desktop
  только потому, что порт занят. Сначала установите причину, используйте другой порт.
- После существенных изменений (новый детектор/env-переменная, поведение Compose,
  новые артефакты или команды запуска) в той же сессии обновляйте AGENTS.md —
  разделы «API и конфигурация», «Детекция», «Проверки и доказательства».

## API и конфигурация

- `POST /process`: запрос `{"payload":"...","payload_id":"..."}`, ответ
  `{"result":"..."}`. Первый запрос маскирует, повтор с полученной маской
  демаскирует через корреляционное хранилище. Для новых тестов используйте новые ID,
  иначе можете измерить кеш/повторный запрос вместо инференса.
- `/health`, `/metrics`, `/docs`; в текущем коде также есть статический `/ui`.
- Пустой `API_KEYS` разрешает `/process` всем; непустой проверяет `X-API-Key`.
  Эта проверка не закрывает автоматически все остальные endpoints.
- `src/core/config.py` по умолчанию задаёт HOST=0.0.0.0, PORT=8000 и DETECTOR=gliner.
  В Compose явно выбран DETECTOR=student, `artifacts/student-pii.pt`.
- Compose берёт `DETECTOR` из окружения хоста: `DETECTOR: "${DETECTOR:-student}"`
  (без переменной поведение прежнее — student).
- Factory знает rules, student, transformer, gliner, hybrid, rubert_onnx. `hybrid`
  сейчас использует GLiNER через factory; это не автоматический гибрид с новым RuBERT.

## Детектор rubert_onnx (notebook-модель rubert-tiny2 в ONNX int8)

- Артефакт: `artifacts/rubert-tiny2-fine-tuning/` (`model_int8.onnx`, `tokenizer.json`,
  `tokenizer_config.json`, `config.json`, `model_config.json` с 53 BIO-тегами,
  max_len=1024, stride=128, public_labels). Скопирован в Docker-образ.
- Код: `src/models/rubert/inference.py` (инференс — точный перенос cell 16 ноутбука
  `train/notebooks/train_pii_masker.ipynb`: окна 1024/stride 128, голосование по
  (start, end), BIO-декод из cell 10), обёртка `src/models/rubert_onnx_detector.py`.
- `PUBLIC_PERSON`/`PUBLIC_ADDRESS` из предсказаний удаляются — это не ПД.
- Включение: env `DETECTOR=rubert_onnx`, путь — `RUBERT_MODEL_PATH`
  (по умолчанию `artifacts/rubert-tiny2-fine-tuning`, в Compose
  `/app/artifacts/rubert-tiny2-fine-tuning`). Запуск:
  `DETECTOR=rubert_onnx docker compose up --build`.
- Score в int8 недетерминирован между батчами (~0.003): спаны/лейблы стабильны,
  тесты сравнивают без score.
- Известные слабости самой модели (не интеграции): пропуск паспорта вида
  «серия 4509 номер 123456» без контекста, обрезание последней цифры номера карты.
- Тесты: `tests/test_rubert_onnx.py` (BIO-декод, детекция, публичные персоны,
  batch vs single, длинные тексты, factory).
- Несколько API-воркеров требуют общего Redis для корреляции; memory не разделяется
  между процессами. `/metrics` хранит счётчики процесса, не агрегирует все воркеры.
- Значения переменных на хосте не переопределяют произвольные жёстко заданные поля
  Compose. Проверяйте `docker compose config`.

## Docker и автоматический публичный адрес

- `docker-compose.yml` теперь запускает localhost.run, а НЕ Cloudflare Quick Tunnel.
- `deploy/tunnel/Dockerfile` собирает небольшой Alpine-образ с OpenSSH/Python;
  `deploy/tunnel/run.py` открывает SSH reverse forwarding `80:api:8000`.
- Туннель ждёт `service_healthy` у API. Healthcheck обращается к внутреннему
  `http://127.0.0.1:8000/health` контейнера API.
- При подключении в stdout появляются `PUBLIC_URL`, `DOCS_URL`, `PROCESS_URL`.
  Из приветственного баннера не извлекаются ссылки документации и QR-коды.
- Текущий адрес находится В КОНТЕЙНЕРЕ в `/run/tunnel/public-url`; это не путь хоста.
  До подключения/после обнаруженного разрыва файл отсутствует. Старые URL остаются
  в истории логов, поэтому для текущей ссылки предпочтительнее файл.
- Автопереподключение — через 5 с; SSH keepalive 20 с, 3 пропуска. Удаление файла
  не означает мгновенное обнаружение любого сетевого обрыва.
- Том `tunnel-state` сохраняет отдельный SSH-ключ и known_hosts. Учётные данные
  пользователя и SSH-agent внутрь не передаются.
- Бесплатный поддомен localhost.run может меняться; стабильность адреса на неделю
  и производительность 2000 RPS не гарантированы. Требуется исходящий SSH на порт 22.
- Redis больше не публикуется на хост: `redis:6379` доступен контейнерам Compose.
  RabbitMQ опционален через profile `rabbitmq`, его порты опубликованы на localhost.
- `API_PORT` меняет только порт хоста, внутренний остаётся 8000. По умолчанию
  API_PORT=8000, API_WORKERS=8. Число воркеров настраивается через окружение.
- `.dockerignore` ограничивает контекст API исходниками, зависимостями и нужными
  runtime-весами; корпус, `.venv`, `.git` и секреты туда не нужны.

```bash
# Запуск с выводом актуальных ссылок в терминал
docker compose up --build
# Фоновый запуск и чтение адреса
docker compose up -d --build
docker compose logs -f tunnel
docker compose exec -T tunnel cat /run/tunnel/public-url
# Если 8000 занят; один воркер для проверки на загруженном ноутбуке
API_PORT=8001 API_WORKERS=1 docker compose up --build
# Остановить публикацию
docker compose stop tunnel
# Локально, без запуска туннеля (уже работающий tunnel сначала остановить)
docker compose up -d --build redis api
```

Последняя проверка: Compose config и извлечение URL проверены, образы собраны.
Полный Compose-прогон НЕ подтверждён: Docker daemon вернул HTTP 500 при проверке
контейнера/образа; ранее наблюдалось завершение дочернего API-процесса при 8 воркерах.
Причина HTTP 500 не установлена. Попытка с одним воркером тоже упёрлась в daemon.
Порты 8000 и 6379 ранее занимали отдельные локальные сервисы — проверять заново.
Отдельно запущенный SSH-туннель localhost.run успешно проксировал `/health` и `/docs`
с HTTP 200. Это не доказательство работоспособности нового Compose или SLA.
Не копируйте старые временные URL из истории как действующие.

## Детекция: измерения и известные проблемы

Полный hybrid dev: 4139 документов, raw detector до public-context postprocessing.
Отчёты: `experiments/pii_recall/baseline_{student,rules}_dev.json`.

| Детектор | Exact-span precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| student | 0.8631 | 0.6656 | 0.7516 |
| rules | 0.2146 | 0.1048 | 0.1409 |

- Это локальные exact typed-span метрики, включая вложенную разметку и только
  annotated_labels каждого источника. Это не балл жюри и не полнота маскирования.
- На этой dev-выборке privacy-character оценка не имела eligible записей:
  нули в её отчёте означают отсутствие измерения.
- Старый char student обрезает до 512 СИМВОЛОВ, старый transformer inference —
  до 256 ТОКЕНОВ. Одна BIO-цепочка не сохраняет вложенные сущности разных типов.
- Старый transformer train использует token accuracy, сохраняет последнюю эпоху,
  ошибается с BIO-границами после пробелов и не учитывает частичную разметку как
  новый эксперимент. Его наличие не означает, что эти проблемы исправлены.
- `src/core/postprocess.py` может исключать ФИО после профессий («врач», «инженер»)
  и адрес после слова «банк». Такая эвристика способна снижать recall; наличие
  слова профессии/банка не доказывает публичность данных.
- Через публичный endpoint наблюдался HTTP 200 с НЕзамаскированным
  `demo@example.com`. Успешный HTTP-ответ не подтверждает качество детекции.
- Старые цифры 600/1800/2490 текстов/с в документации не доказывают 2000 RPS
  на текущей модели, машине и публичном туннеле.

## Новый RuBERT-эксперимент

- Исследование и детали: `experiments/pii_recall/README.md`.
- Обучение: `experiments/pii_recall/train.py`; Kaggle runner:
  `scripts/kaggle_recall_train.py`; упаковка: `scripts/push_kaggle_recall.py`;
  удобная команда: `scripts/push_kaggle_rubert.sh`.
- Основа: `redmadrobot-rnd/rubert-base-pii-ner`. Дообучается encoder, исходная
  43-классовая BIO-голова заменена независимыми O/B/I heads по типам проекта.
- Частично размеченные классы и padding/special tokens исключены из loss.
  Разные типы могут перекрываться; вложенность одного типа отдельно не решена.
- Gold hybrid train без автоматического добавления teacher pseudo-labels.
  Проверяются совпадения нормализованного текста между split. Это не исчерпывающая
  проверка всех семейств шаблонов и утечек сущностей.
- Train/dev/test: 37364 / 4139 / 4163 записи; train источники — FactRuEval, NEREL,
  redmadrobot и синтетическое дополнение. Публичное происхождение не следует
  трактовать как отсутствие любых ПД в исходных текстах.
- Dev выбирает checkpoint и общий bias PII-логитов; precision floor=0.90,
  target recall=0.95. Test оценивается после выбора. Проверяются метрики по типам
  и `required_labels_all_pass`; отсутствие support не считается успехом.
- Нативный `max_position_embeddings` RuBERT — 512. Новый wrapper: 1024 токена
  вместе со special tokens, stride=256, 10 эпох, batch=2, accumulation=4,
  LR=2e-5, FP16 mixed precision, gradient checkpointing.
- Расширение BERT позиций явное: первые 512 сохранены, новые инициализируются
  повторением исходных; обновлены buffers, config и tokenizer. Это не делает
  модель автоматически качественной на длинном контексте: нужны обучение и оценка.
- Большие документы обрабатываются перекрывающимися окнами. 1024 — размер окна,
  не размер всего допустимого документа; производительность на 100k не подтверждена.
- FP8 не реализован. На T4/P100 нет нужного аппаратного FP8; не называйте FP16
  обучением в FP8 и не путайте INT8-квантизацию с FP8.
- Сохраняются encoder/tokenizer, `head.pt`, `experiment.json`, dev/test metrics
  и training log; настройки содержат corpus SHA256 и revision encoder.
- Новый формат НЕ совместим с прежним `TRANSFORMER_MODEL_PATH` без отдельного
  inference-адаптера. В Docker по-прежнему работает student, не новый RuBERT.

## Kaggle: запуск и ограничения

Пользователь явно разрешил загрузку hybrid train/dev/test в приватный
`theodaimones888/ru-pii-recall-base-v1-bundle` и запуск обучения. Не спрашивайте
снова об уже разрешённом действии. Предпочитайте повторное использование этого
датасета: передаётся только код/конфигурация, без повторной загрузки корпуса.

```bash
bash scripts/push_kaggle_rubert.sh \
  --reuse-dataset theodaimones888/ru-pii-recall-base-v1-bundle --submit
.venv/bin/kaggle kernels status theodaimones888/ru-pii-recall-base-1024-v1
.venv/bin/kaggle kernels output theodaimones888/ru-pii-recall-base-1024-v1 \
  -p artifacts/recall-rubert-1024
```

- Без `--submit` скрипт только собирает бандл. Без `--reuse-dataset` создаётся
  отдельный датасет; текущая автоматическая проверка ранее отклоняла такой новый
  адрес назначения. Не обходите отказ: код-only reuse уже использовался успешно.
- В reuse-режиме runner переопределяет старую конфигурацию датасета и пишет текущие
  исходники в `/kaggle/working/recall-code`; корпус берётся из существующего input.
- `ru-pii-recall-base-1024-v1`, version 1, был успешно отправлен и возвращал RUNNING.
  Завершение, результаты test и достижение 0.95 ещё не проверены.
- Предыдущий `ru-pii-recall-base-v1` запускался с окном 384, 10 эпохами и FP16.
  Позднее status вернул 403; это НЕ подтверждение остановки.
- Доступный CLI не имеет `kernels cancel`. SDK cancel требует kernel_session_id,
  которого status/get-kernel не возвращают. kernel_id НЕ является session_id.
  Не подставляйте выдуманный ID и не удаляйте kernel ради попытки остановки.
- Установленный CLI игнорирует поле machine_shape при push: не обещайте 2×T4.
  Runner использует один GPU и выводит его фактическое имя/capability в лог.

## Проверки и доказательства

```bash
docker compose config --quiet
.venv/bin/python -m pytest tests/test_recall_experiment.py -q
.venv/bin/python -m pytest tests/test_rubert_onnx.py -q
.venv/bin/python scripts/evaluate_pii_recall.py --detector student \
  --out experiments/pii_recall/baseline_student_dev.json
```

- В сессии прошли 8 тестов нового эксперимента: partial/nested BIO, декодирование,
  утечка split, отсутствие support, оконный хвост, BERT forward/backward на 1024,
  save/reload, reuse-бандл без корпуса. Это не тест всей системы и не замер качества.
- Локальный smoke на маленьком случайном BERT проверил train/save/reload/test,
  gradient checkpointing и накопление. Его метрики не характеризуют RuBERT.
- Для SLA измеряйте завершённые успешные запросы в секунду, p95/p99, ошибки,
  размер payload, тип операции, параметры клиента и длительность пика.
  Проверяйте отдельно локальный API и публичный маршрут; GET /health не заменяет
  POST /process с реальным инференсом. Не объявляйте SLA по числу открытых sockets.

## Первичные источники (сведения о тарифах перепроверять)

- RuBERT: https://huggingface.co/redmadrobot-rnd/rubert-base-pii-ner
  В model card exact recall=0.855; F1 около 0.95 относится к другому протоколу
  PERSON+LOCATION overlap, не ко всем типам ТЗ.
- localhost.run: https://localhost.run/docs/forever-free/
  Бесплатный адрес меняется, есть ограничение скорости; не гарантия 2000 RPS.
- SSH CLI: https://localhost.run/docs/cli/
- Kaggle cancellation: https://github.com/Kaggle/kaggle-cli/issues/1169
- Cloudflare Quick Tunnel: https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/
  Ограничение 200 in-flight относится к Quick Tunnel, не ко всем Cloudflare Tunnel.
- Постоянный Cloudflare Tunnel: https://developers.cloudflare.com/tunnel/get-started/
  Для публикации нужен домен на Cloudflare. IP телефона не нужно вводить в A-record
  для такого туннеля. Сам постоянный Tunnel в этом проекте не настроен.
- Pinggy: https://pinggy.io/ — бесплатная сессия 60 минут; вариант Pro/trial
  пользователь отверг из-за запроса карты. Больше не предлагайте его как готовое решение.

## Контекст параллельного трека из origin/main

Следующий раздел сохранён при объединении с a4808aa. Описания `train/`, CSV и
notebook относятся к отдельному треку rubert-tiny2, а не к RuBERT-эксперименту
выше. Статусы запусков, файлов и LFS — исторические; проверять по текущему Git
и окружению. При расхождении настроек runtime сверяться с кодом и Compose.


Репозиторий хакатонного проекта «Модуль безопасности персональных данных» (ТЗ — `Hackathon task.md`).
Здесь зафиксирован контекст проекта, ключевые решения и команды, чтобы работать без повторного исследования.

## Задача

**Модуль безопасности персональных данных** Банка: сервис-прокси Core-компонент в цепочке
«система-потребитель — LLM». Обрабатывает запрос, отправляемый в LLM, и ответ LLM:
идентификация типов ПД, маскирование, демаскирование. Должен отвечать быстро, качественно
и выдерживать нагрузку. Контекст: продуктов с LLM в Банке много, каждая команда защищает ПД
по-своему — нужно единое переиспользуемое решение (соответствие законодательству РФ и требованиям ЦБ).

### Контракт `/process` и автоматическая проверка

- Единственный эндпоинт `POST /process`: запрос `{payload, payload_id}` → ответ `{result}`.
- Первый запрос с новым `payload_id` — маскирование (вернуть маску, запомнить соответствие),
  повторный с тем же — демаскирование (вернуть исходную строку). Идемпотентность по `payload_id`
  обязательна: ретраи до 2 на каждый запрос (итого до 3 попыток).
- Проверяющая система (AlfaSonar) гоняет пары по эталонному датасету: прямая проверка — сравнение
  с эталонной маской (нормированный span-based Левенштейн [0..1]), обратная — сравнение
  с исходной строкой. Категории: ФИО, даты, паспорт, гражданство, орган выдачи, код подразделения,
  в/у, адрес, email, телефон, ИНН, номер карты, CVV, ПИН, имя держателя, сложные предложения.
- Нагрузочный прогон: ~5 минут по всем категориям, целевой RPS 1000, элементы датасета
  переиспользуются; таймаут одного запроса 10 с; 429 допустим (учитывается `Retry-After`),
  стоп после 5 невалидных запросов подряд; целевая latency ≤ 1 c.
- Обе проверки обязательны: (1) zip только с исходным кодом — без датасетов, зависимостей,
  build-каталогов (проверка качества кода); (2) развёрнутый доступный сервис по контракту —
  http или https (self-signed допускается, проверка идёт с `ssl_verify=false`).
- Эталонный эндпоинт `https://process-test.holydev.space/process` — stateless эхо, показывает
  только формат обмена; правил маскирования не раскрывает. Единственный источник правил —
  пример из OpenAPI ТЗ: `Иванов Иван Иванович → И. И. И.`, `4509 123456 → 45** ****56`.
- SLA инференса: 1000 RPS / latency < 1 c, CPU-only 4 ядра / 8 ГБ (обучение — на GPU отдельно).

### Типы ПД (ТЗ п. 4.1)

16 пунктов ТЗ: ФИО; дата рождения; место рождения; серия и номер паспорта; гражданство;
орган, выдавший паспорт; код подразделения; дата выдачи паспорта; серия и номер водительского
удостоверения; адрес (целиком и отдельно: страна, индекс, город, улица, дом, квартира); email;
номер телефона; номер платёжной карты; пин-код; ИНН; CVV-код; имя держателя карты.
В нашей таксономии после декомпозиции (адрес → части, паспорт/ВУ → серия+номер): **24 метки ПД**
+ `PUBLIC_PERSON`/`PUBLIC_ADDRESS` = 26 BIO-меток → 53 класса.

Особые требования: принадлежность именно к персональным данным (Пушкин — не ПД, адрес отделения
Банка — не ПД); регистронезависимость; возможность расширения списка типов; логирование
выявленных типов ПД по каждому запросу.

### Маскирование и демаскирование (ТЗ п. 4.2–4.3)

- Маскирование должно учитывать вариации написания: регистр; разный порядок частей даты
  (мм.дд.гггг / гггг.дд.мм); дата текстом; разделяющие слова («серия хххх номер хххххх»).
- Демаскирование — восстановление с сохранением местоположения знаков.

### Общие и нефункциональные требования (ТЗ п. 4.4, 5)

- Качество идентификации/маскирования/демаскирования — 95%; тексты до 100 000 токенов.
- Логирование каждого этапа; метрики Latency / RPS / TPS; исходные ПД не попадают в логи и метрики.
- Ограниченный список систем-потребителей; гибкая настройка по системам: вкл/откл обращения,
  перечень типов ПД для маскирования, наличие демаскирования; краткая инструкция по настройке
  (≤ 5 предложений).
- Безопасность: защищённый контур, шифрование, минимизация данных. Надёжность: корректная
  обработка ошибок, деградация функционала при недоступности компонентов. Масштабируемость:
  новые типы ПД и рост RPS без переделки сервиса.

### Дополнительные плюсы (ТЗ п. 6)

Токенизация/детокенизация или замена синтетическими данными; настройка вида маскирования
по системам; RPS 2000 при latency ≤ 1 c; документы, удостоверяющие личность, кроме паспорта РФ;
маскирование только при нескольких однозначно идентифицированных типах ПД («пин-код» один —
не маскируем, «пин-код + номер карты» — маскируем), с возможностью настройки.

### Что оценивают («нам важно увидеть»)

Находит ли всё, что нельзя передавать без обработки; мало ли ложных срабатываний; сохраняется ли
смысл запроса для LLM после обработки; можно ли управлять правилами для разных систем; простота
подключения нового потребителя; поведение при ошибках и недоступности компонентов; не попадают ли
исходные защищаемые данные в логи и технические метрики.

## Два трека решения

1. **rubert-tiny2 BIO-теггер** — обучение в `train/` (окружение и команды — в разделе «Окружение обучения (train/)»).
2. **GLiNER + дистилляция** (смержено из origin/main): fine-tune GLiNER в `ru_gliner_hybrid_v4/`
   (base-small → gliner-ru-pii-small), дистиллированный student-детектор `artifacts/student-pii.pt`,
   обучение на Kaggle 2xT4 — `scripts/kaggle_*`, сервис с переключаемыми детекторами — `src/`.

## Ключевые решения (проверены, менять только с пониманием последствий)

- Детектор: `cointegrated/rubert-tiny2` — 3 слоя, hidden 312, vocab 83 828, **max_pos 2048** (не 512). BIO-теггинг: 26 меток → 53 класса.
- 26 меток = 24 типа ПД из ТЗ + `PUBLIC_PERSON` / `PUBLIC_ADDRESS` (публичные персоны и адреса НЕ маскируются; в данных это точные близнецы `PERSON==PUBLIC_PERSON`, `ADDRESS==PUBLIC_ADDRESS`).
- Hard negatives двух видов: (а) переметка классиков в factrueval/nerel — `scripts/relabel_public_persons.py` добавляет `PUBLIC_PERSON`-близнецы по стемам фамилий из `scripts/public_persons.py` (распространённые фамилии и современные политики в исключениях — RELABEL_EXCLUDED); (б) сгенерированная добавка `data/public_negatives/` — 143 известные персоны в ~48 шаблонах (чистые негативы, смешанные «публичная персона + реальные ПД», адреса отделений банков), регистровые варианты keep/lower/upper 60/25/15.
- Приоритет тегов при вложенности: `PUBLIC_*` побеждает всё → `BIRTH_PLACE` (близнецы `CITY/COUNTRY`) → innermost-wins (`CITY/STREET/HOUSE` вместо контейнера `ADDRESS`; контейнер всегда полностью выводится из частей).
- Канонизация gold-спанов идёт тем же BIO-путём (tokenize → labels → decode), что и предсказания — самосогласованно. Спаны расширяются до границ wordpiece-токенов (например «№», приклеенный к номеру карты).
- Инференс: чанки max_len 1024 / stride 128 → голосование по (start, end) ключам токенов между чанками → BIO-декод → char-спаны.
- Метрики детекции: span-F1 (exact match тройки start/end/label), char-coverage P/R/F1, FP на `PUBLIC_*`. Итоговая метрика ТЗ (Левенштейн по маскам) считается на уровне сервиса после подключения правил маскирования — от модели зависит только детекция.
- Демаск по контракту: хранение `payload_id → оригинал` (100% по построению).

## Структура репозитория

| Путь | Что это |
|---|---|
| `Hackathon task.md`, `ds.pdf` | ТЗ хакатона — источник всех требований |
| `rules/` | критерии оценивания хакатона + черновики |
| `train/` | изолированное окружение обучения rubert-tiny2: notebook, Dockerfile (GPU, порт 9132, корп. сертификаты), requirements, certs — см. раздел «Окружение обучения (train/)» |
| `data/csv/{train,dev,test}.csv` | датасет для обучения: 91 713 / 10 961 / 11 044 записей, 251k / 30.1k / 30.4k сущностей; колонки `id, split, source, text, entities` (compact JSON, char-оффсеты) |
| `data/hybrid/` | исходный корпус rubert-трека: factrueval2016 + nerel + redmadrobot_pii_train + task-synthetic-supplement-v2; `manifest.json` — эталон счётчиков |
| `data/wolframko/` | интегрированный `wolframko/russian-pii-66k` (65 652 записи, Presidio-метки → таксономия ТЗ); `manifest.json` |
| `data/public_negatives/` | сгенерированные hard negatives (2 400 строк: 1 300 чистых, 788 смешанных, 312 банковских адресов); `manifest.json` |
| `data/real/` | real-корпус GLiNER-трека (nerel + factrueval), собран пайплайном `ru_gliner_hybrid_v4` |
| `data/supplement/` | синтетический суплемент GLiNER-трека (12k / 1.5k / 1.5k, seed 20260922) |
| `data/redmadrobot/`, `data/third_party/` | исходники redmadrobot_pii_train до конверсии |
| `scripts/` | конвейер данных rubert-трека: `public_persons.py`, `make_public_negatives.py`, `relabel_public_persons.py`, `jsonl_to_csv.py`, `integrate_wolframko.py`, `build_train_notebook.py` |
| `scripts/kaggle_*.py`, `scripts/push_kaggle_*.sh` | обучение GLiNER/student на Kaggle 2xT4 (GLiNER-трек) |
| `src/` | сервис: FastAPI `/process` (`api`), worker + Redis + RabbitMQ (`core`), детекторы rules/gliner/hybrid/student (`models`) |
| `mirror/` | сервис-зеркало `/process` для наблюдения за трафиком: эхо payload, эмуляция идемпотентности по `payload_id`, лог всех тел запросов в `logs/requests.log` (JSONL); свой Dockerfile (порт 8010, `LOG_PATH` для volume) |
| `ru_gliner_hybrid_v4/` | GLiNER-проект: пакет `ru_pii`, модели base-small и gliner-ru-pii-small (веса через LFS), notebooks, configs, reports, tests |
| `artifacts/student-pii.pt` | дистиллированный student-детектор (GLiNER-трек) |
| корневые `Dockerfile`, `requirements.txt`, `docker-compose.yml` | **runtime-образ сервиса** (не обучения): python:3.10-slim, копирует src, ru_pii, gliner-модель, student-чекпоинт |

## Сервис-зеркало (mirror/)

Заглушка контракта `/process` для наблюдения за реальным трафиком проверяющей системы.
Не содержит логики маскирования: эхо payload, повторный `payload_id` → возврат сохранённого
исходника (эмуляция демаскирования в памяти процесса). Каждое тело запроса пишется в JSONL
`mirror/logs/requests.log` (`ts`, `payload_id`, `payload`) — запись асинхронная
(`asyncio.to_thread` + `asyncio.Lock`), event loop не блокируется.

Запуск локально: `python3 -m uvicorn mirror.app:app --host 0.0.0.0 --port 8010`
(env: `PORT` не используется uvicorn-CLI, порт задаётся флагом; `LOG_PATH` — путь к логу).

Docker: `docker build -t pii-mirror -f mirror/Dockerfile .`; запуск с volume для логов:

```bash
docker run -d --rm --name pii-mirror -p 8010:8010 --user $(id -u):$(id -g) \
  -v $(pwd)/mirror/logs:/app/logs -e LOG_PATH=/app/logs/requests.log pii-mirror
```

(контейнер работает от `nobody`, volume монтируется с `--user` хост-пользователя для прав записи;
базовый образ `python:3.12-slim` — 3.10-slim недоступен из-за таймаутов registry).

## Окружение обучения (train/)

Изолированное окружение для обучения детектора ПД (трек rubert-tiny2). Папка переносится
на GPU-сервер / в k8s-под целиком, вместе с `data/csv/` из корня репо (~60 МБ).
Маскирование/демаскирование — логика сервиса, в notebook не входят.

### Состав

| Файл | Что это |
|---|---|
| `train/notebooks/train_pii_masker.ipynb` | детектор: BIO-теггинг rubert-tiny2, class weights, кастомный torch-луп (AdamW 3e-4, warmup+cosine, лучший чекпоинт по dev macro span-F1), инференс с чанкингом, метрики, error analysis, ONNX int8 export |
| `train/Dockerfile` | базовый образ `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime`, порт 9132, non-root `appuser` (UID/GID 1001), VimpelCom CA-сертификаты вшиты (сборка и pip за корпоративным прокси), pip `--timeout 300 --retries 10` |
| `train/requirements.txt` | torch НЕ указан — идёт с базовым образом. transformers 5.14.1, onnx/onnxruntime, jupyterlab 4.4.4, gliner[training]==0.2.29 (эксперименты GLiNER-трека), datasets, scikit-learn, pyarrow |
| `train/certs/` | корпоративные root/int CA + ca-bundle — нужны Dockerfile при сборке за прокси, не удалять |

### Правила

- Параметры notebook'а задаются env-переменными (не правкой файла): `PII_DATA_DIR`, `PII_OUT_DIR`,
  `PII_MODEL_NAME` (например, `PII_MODEL_NAME=models/rubert-tiny2` — локальная копия модели, без скачивания с HF).
- Данные ищутся автоматически: `data/csv/train.csv` или `csv/train.csv` вверх по дереву от cwd —
  работает и в репозитории, и в поде.
- Данные в Docker-образ не запекаются — копируются в под отдельно.

### Команды

```bash
docker build -t pii-masker-notebook train/
docker run -d --gpus all -p 9132:9132 -e JUPYTER_TOKEN=секрет pii-masker-notebook
```

Перенос CSV в под ai-platform-jupyter-lab (в поде данные — `/workspace/models/alfa_hack/csv`):

```bash
kubectl exec <pod> -- mkdir -p /workspace/models/alfa_hack/csv
tar -C ./csv -cf - . | pv -btpre | kubectl exec -i <pod> -- tar xf - -C /workspace/models/alfa_hack/csv
```

### Результат обучения

`model_piinet/` (лучший чекпоинт) + ONNX int8 создаются рядом с данными
(по умолчанию — родитель каталога с CSV) и забираются в сервис.

## Команды

```bash
python3 scripts/jsonl_to_csv.py            # пересобрать data/csv из hybrid + wolframko + public_negatives
python3 scripts/integrate_wolframko.py     # переинтегрировать wolframko (idempotent, parquet кэшируется в data/wolframko/raw/)
python3 scripts/make_public_negatives.py   # перегенерировать hard negatives (seed, idempotent)
python3 scripts/relabel_public_persons.py  # переметка классиков в hybrid (перед этим НЕ пересобирать hybrid!)
python3 scripts/build_train_notebook.py    # перегенерировать train/notebooks/train_pii_masker.ipynb

# сервис (GLiNER-трек)
DETECTOR=rules CORRELATION_STORE=redis API_WORKERS=4 python -m src.run_api
docker compose up

# LFS-веса gliner (нужен доступный github.com)
git lfs pull
```

Проверка целостности данных после пересборки CSV: количество строк против манифестов, `text[start:end] == entity.text` для всех спанов (см. алгоритм в `scripts/jsonl_to_csv.py`).

## Конвенции

- Валидация на границах обязательна: оффсеты, счётчики против `manifest.json`, уникальность id, запрет crossing-спанов. Ошибка валидации — громкий fail, не тихий скип.
- Notebook правится только через `scripts/build_train_notebook.py` (ручные правки затираются при регенерации); после изменения состава данных синхронизировать `EXPECTED_COUNTS` в builder'е с фактическими счётчиками CSV.
- Скрипты конвейера данных — stdlib-only (pandas/pyarrow только для parquet).
- Язык коммуникации и документации — русский.
- Данные не коммитятся и не попадают в архив решения (требование ТЗ п. 7.1).
- Сетевые особенности: PyPI может таймаутиться (в train/Dockerfile решено сертификатами + `--timeout 300 --retries 10`), huggingface.co обычно доступен, github.com из рабочего окружения может таймаутиться (LFS-веса тянуть при доступной сети).
- Git: `artifacts/student-pii.pt` закоммичен в origin/main мимо LFS и из-за `*.pt → lfs` в `.gitattributes` вечно показывает ` M` — не коммитить его.

## Текущее состояние

- Выполнен merge origin/main (3af5b9f, GLiNER-трек + сервис src/). Вся локальная работа (`data/`, `scripts/`, `train/`, `AGENTS.md`, `Hackathon task.md`) — untracked поверх main, история origin/main не переписана.
- LFS-веса `ru_gliner_hybrid_v4/models/*/model.safetensors` (2 × 664 МБ) — заглушки-указатели; при появлении сети к github.com выполнить `git lfs pull`.
- Notebook синхронизирован с builder'ом: автопоиск данных (`data/csv` или `csv` вверх по cwd), env-параметры `PII_DATA_DIR` / `PII_OUT_DIR` / `PII_MODEL_NAME`. Smoke-тест пройден ранее (все ячейки, чанкинг, unit-тесты BIO, round-trip, ONNX export), полное обучение — на GPU.
- CSV передаются в под ai-platform-jupyter-lab (kubectl, `/workspace/models/alfa_hack/csv`) — команды в разделе «Окружение обучения (train/)».

## Дальнейшие шаги

1. Обучить rubert-tiny2 в поде (train/), забрать `model_piinet/` + ONNX int8.
2. `git lfs pull` — забрать GLiNER-веса из origin/main.
3. Сервис по контракту `/process`: выбор детектора (наш rubert / student / gliner / hybrid / rules), хранилище `payload_id → оригинал`, правила маскирования из примера ТЗ (словарь, конфигурируемо).
4. Нагрузочные цели: 1000 RPS / 4 CPU — ONNX int8 + кэш по хэшу payload.

## Рефакторинг слоя детекторов

- `src/models/factory.py`: реестр конструкторов (Factory/Strategy), явные
  `config: Settings` и `builders` для внедрения зависимостей. Переданный реестр
  заменяет стандартный; старый вызов `create_detector()` работает как прежде.
- PostProcessedDetector — Decorator; фабрика применяет его один раз снаружи
  готового детектора. Стандартный hybrid строится на GLiNER без внутреннего фильтра.
- `src/models/batched.py`: Template Method для student/transformer inference.
  Подклассы реализуют `_predict_nonempty`; общий код сохраняет порядок, пропускает
  пустые строки и проверяет количество результатов. Пустой батч возвращает `[]`.
- Проверки: `tests/test_detector_composition.py` — фабрика, композиция hybrid,
  пустые/смешанные батчи, реальный маленький student без загрузки весов/сети.
- Эти изменения не меняют формат checkpoint, BIO-декодирование, ограничения окна
  или запущенный Kaggle-эксперимент; измерений ускорения и качества здесь нет.
