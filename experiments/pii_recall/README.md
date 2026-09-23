# Исследование recall русских ПД — 23.09.2026

Цель: измерить и поднять recall до 0.95, сохранив precision; достижение цели пока **не подтверждено**. Новые эксперименты изолированы от текущего API и пользовательских незакоммиченных изменений.

## Требования из rules

`ds (1).pdf`, стр. 2–4: 17 групп ПД, включая ФИО, место/дату рождения, гражданство, реквизиты паспорта и ВУ, части адреса, email/телефон/ИНН/карту/CVV/PIN/держателя карты. Нужны вариации регистра, текстовые даты, разделяющие слова, исключения для публичных упоминаний. Текст до 100 000 токенов; 1000 RPS и latency ≤1 с. В ТЗ «качество 95%» не определено как конкретная метрика; recall ≥0.95 — дополнительная цель пользователя. Критерии оценивают также ложные срабатывания, обратимость и поведение API.

Предлагаемый контроль: exact typed-span micro precision/recall/F1, recall и support каждого обязательного типа, полнота маскирования символов и доля запросов с утечкой на privacy-размеченной выборке. Порог precision 0.90 в эксперименте — инженерная настройка, а не число из ТЗ. Прохождение micro recall не заменяет прохождение всех типов. Вложенные coarse/fine метки делают строгий exact-span протокол строже возможного протокола организаторов.

## Что мешает сейчас

- Docker использует char BiLSTM-CRF (`student`), который обрезает вход на 512 **символах**. Transformer обрезает на 256 токенах. Это гарантированные пропуски хвоста длинного текста.
- `src/models/transformer/train.py` считает token accuracy, не recall; сохраняет последнюю эпоху. Преобладание O-токенов может скрывать низкую полноту.
- BIO-конвертация проверяет предыдущий символ: пробел между словами одного ФИО ошибочно создаёт новый B-тег.
- Корпус частично размечен: отсутствие разметки класса не означает отсутствие его сущностей. Старый loss учит такие случаи как O.
- В train 37 364 записи, dev 4 139, test 4 163. Есть вложенные сущности (паспорт и его части, адрес и компоненты). Одна BIO-цепочка затирает часть разметки.
- Обычный NER-корпус новостей не обучает различать персональные/публичные упоминания. PUBLIC_* не следует превращать в безусловный whitelist.
- Текущий `filter_public` может удалить ФИО клиента после слов «инженер», «врач», «программист»; встреча слова «банк» в предыдущих 60 символах тоже не доказывает публичность адреса. Это отдельный источник падения recall после модели.
- Числа 600/1800 текстов в секунду для transformer в старых docstring не являются воспроизводимым нагрузочным тестом.

## Измеренная базовая линия

Полный hybrid dev, 4 139 документов; raw detector до public-context postprocessing. Exact typed spans учитывают вложенные метки и только классы, размеченные в источнике.

| Детектор | Precision | Recall | F1 |
|---|---:|---:|---:|
| rules | 0.2146 | 0.1048 | 0.1409 |
| student (текущий checkpoint) | 0.8631 | 0.6656 | 0.7516 |

Rules-only закономерно не покрывает семантические классы; низкая precision дополнительно требует разбора контекстных правил и соответствия границ разметке. Это локальный протокол, не оценка жюри. Полные отчёты: `baseline_rules_dev.json`, `baseline_student_dev.json`. Для student это 9 772 точных совпадения, 4 909 пропущенных сущностей и 1 550 лишних предсказаний. До цели recall 0.95 остаётся существенный разрыв.

Самые слабые обязательные типы student: PASSPORT_NUMBER 0.0685, ADDRESS 0.1374, DRIVER_LICENSE_SERIES 0.1565, CITY 0.1989, PASSPORT_SERIES 0.2958; PERSON 0.7686. Часть провалов связана с невозможностью одной BIO-цепочки воспроизвести вложенные метки. В текущем dev нет записей, допущенных к privacy-character оценке: нули в соответствующем разделе отчёта означают отсутствие измерения, а не нулевую полноту маскирования.

## Кандидаты

| Вариант | Зачем | Ограничение |
|---|---|---|
| ruBERT-tiny2 + независимый BIO-head для каждого типа | Быстрая базовая линия с вложенными сущностями и корректным loss | 29.4M параметров; качество нужно измерить |
| redmadrobot-rnd/rubert-base-pii-ner encoder + новые BIO-heads | Русскоязычный encoder уже адаптирован к ПД; первый кандидат на качество | Опубликованный recall готовой модели 0.855 exact; старая голова не покрывает все типы ТЗ |
| microsoft/mdeberta-v3-base + те же heads | Альтернативный multilingual encoder, полезен для смешанного RU/EN | Более дорогой inference; запускать после первых двух |
| GLiNER multilingual / дообученный существующий GLiNER | Span-модель естественно поддерживает вложенность | Калибровка порогов и скорость; нет доказанного R≥0.95 для нашего ТЗ |

Источники:
- [ruBERT-tiny2](https://huggingface.co/cointegrated/rubert-tiny2): русский encoder, 29.4M параметров, downstream fine-tuning.
- [Russian PII NER](https://huggingface.co/redmadrobot-rnd/rubert-base-pii-ner): exact P/R/F1 0.819/0.855/0.836; pipeline 0.904/0.875/0.889. Число 0.95 на другой таблице — F1 с overlap matching только PERSON+LOCATION, не recall всех типов.
- [mDeBERTa](https://huggingface.co/microsoft/mdeberta-v3-base): альтернативный pretrained backbone.
- [GLiNER multi PII](https://huggingface.co/urchade/gliner_multi_pii-v1): PII finetune, не готовая гарантия русского банковского качества.

Рекомендация — сначала исправленный tiny против PII-adapted RuBERT; затем сравнить нейросеть и гибрид с контекстными правилами для структурных типов. Дистилляция имеет смысл после подтверждения качества teacher. Добавление предсказаний teacher в gold само по себе не гарантирует улучшения, особенно при смене политики разметки.

## Новый эксперимент

`train.py`: отдельные 3-классовые O/B/I heads на каждый тип. Разные типы могут перекрываться; неизвестные классы и special/padding tokens исключены из loss. Окна с перекрытием охватывают весь текст; inference объединяет токены по абсолютным offset, предпочитая центральное окно. Веса encoder, tokenizer, head и настройки сохраняются для offline загрузки. Это прототип; скорость на 100k токенах и SLA не подтверждены.

Обучение: gold hybrid train без teacher pseudo-labels; проверка совпадений нормализованного текста между split. Dev выбирает эпоху и общий bias PII-логитов из {0,0.5,1,1.5,2}, предпочитая precision≥0.90 и recall≥0.95; если достижимого порога нет, сохраняет лучший доступный результат без объявления успеха. Test оценивается один раз после выбора. Отчёт содержит метрики каждого типа и `required_labels_all_pass`; недостаток support означает непрохождение. Сравнение нескольких архитектур всё равно требует финального независимого holdout перед заявлением качества.

Не решено автоматически: вложенные сущности **одного** типа, точные границы внутри subword, privacy-классификация и интеграция в API. Synthetic dev/test проверяют только имеющиеся семейства сценариев, а не скрытый датасет жюри. Нужны независимые банковские контрпримеры, текстовые даты, регистр, опечатки, длинные документы и одновременное присутствие публичных и частных данных. Метрики на частичной разметке учитывают только размеченные классы и не доказывают precision на полностью размеченном продовом потоке.

## Запуск

```bash
# Сборка без передачи данных
.venv/bin/python scripts/push_kaggle_recall.py --model tiny --out /tmp/pii-recall-tiny-bundle
.venv/bin/python scripts/push_kaggle_recall.py --model base --out /tmp/pii-recall-base-bundle

# Создать приватные датасеты и запустить GPU kernels
.venv/bin/python scripts/push_kaggle_recall.py --model tiny --submit
.venv/bin/python scripts/push_kaggle_recall.py --model base --submit

# Контроль и получение результатов после успешного запуска
.venv/bin/kaggle kernels status theodaimones888/ru-pii-recall-tiny-v1
.venv/bin/kaggle kernels output theodaimones888/ru-pii-recall-tiny-v1 -p artifacts/recall-tiny

# Базовая линия без обучения
.venv/bin/python scripts/evaluate_pii_recall.py --detector student --out experiments/pii_recall/baseline_student_dev.json
.venv/bin/python -m pytest tests/test_recall_experiment.py -q
```

Upload включает только новый training script, schema/metrics, train/dev/test и конфигурацию. Ключи, локальные веса и остальной workspace не включаются. Kaggle runner использует один GPU, не обещает ускорение 2×T4. Скрипт не перезаписывает существующий датасет автоматически. Версии torch, revision encoder и SHA256 исходных split сохраняются в experiment.json.

Локальная проверка: 5 unit-тестов и полный локальный smoke (случайный маленький BERT: train/save/reload/test) прошли. Smoke подтверждает работоспособность кода, не качество. Приватный RuBERT-прогон отправлен после явного разрешения пользователя; см. актуальный запуск ниже.

## Fine-tuning redmadrobot RuBERT

```bash
# Подготовка приватного бандла, без загрузки
bash scripts/push_kaggle_rubert.sh --out /tmp/pii-rubert-bundle

# Подготовка, загрузка приватного датасета и отправка GPU-задачи
bash scripts/push_kaggle_rubert.sh --submit

# Настройки можно переопределить
bash scripts/push_kaggle_rubert.sh --epochs 8 --batch-size 8 --lr 2e-5 --submit

# Проверка статуса и скачивание после завершения
.venv/bin/kaggle kernels status theodaimones888/ru-pii-recall-base-v1
.venv/bin/kaggle kernels output theodaimones888/ru-pii-recall-base-v1 -p artifacts/recall-rubert
```

Нужен настроенный Kaggle CLI (`~/.kaggle/kaggle.json` или поддерживаемые CLI переменные окружения). Ключ в бандл не попадает. `PYTHON_BIN` позволяет выбрать Python вместо `.venv/bin/python`.

Параметры RuBERT-wrapper по умолчанию: 10 эпох, batch 8, LR 2e-5, max_length 384, stride 96, FP16 с GradScaler на CUDA. Дообучается весь pretrained encoder `redmadrobot-rnd/rubert-base-pii-ner`; исходный 43-классовый BIO classifier не переносится, вместо него обучаются независимые O/B/I heads по каноническим типам проекта. Это позволяет учитывать вложенность разных типов и частично размеченные источники. На CPU FP16 автоматически отключается.

Результат: `recall-model.tar.gz`, внутри `encoder/` с tokenizer, `head.pt`, `experiment.json`, `dev_metrics.json`, `test_metrics.json`. Это новый формат модели; старый `TRANSFORMER_MODEL_PATH` не умеет его читать без адаптера. Достигнутый recall и precision нужно смотреть в test_metrics.json; обучение само по себе не доказывает recall≥0.95.

Передача hybrid train/dev/test на Kaggle разрешена пользователем. Приватный датасет создан и имеет статус ready; kernel version 1 успешно отправлен.


Обновление запуска: пользователь явно разрешил передачу указанного корпуса на Kaggle; загрузка приватного датасета принята к исполнению. Лимит RuBERT увеличен до 10 эпох, лучший checkpoint по-прежнему выбирается на dev. FP8 для стандартных T4/P100 не поддерживается; этот запуск использует FP16 mixed precision, а GPU и compute capability печатаются в лог. Источник: [NVIDIA Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/). FP8-веса не создаются и FP8-ускорение не заявляется.

Предыдущий запуск (384 токена): [ru-pii-recall-base-v1](https://www.kaggle.com/code/theodaimones888/ru-pii-recall-base-v1), version 1, 10 эпох, FP16; датасет `theodaimones888/ru-pii-recall-base-v1-bundle`, приватный. Результаты качества появятся после завершения обучения.


## Контекст 1024 токена — подготовлен следующий прогон

В публичном config redmadrobot RuBERT `max_position_embeddings=512`. Предыдущий запуск использовал окна 384/96. Большие документы и раньше обрабатывались перекрывающимися окнами целиком: размер окна не является лимитом документа.

Теперь `scripts/push_kaggle_rubert.sh` по умолчанию задаёт:

- 1024 токена в окне (включая special tokens), overlap 256;
- 10 эпох, FP16, batch 2, накопление градиентов 4 (эффективный batch 8 окон);
- gradient checkpointing;
- явное расширение абсолютных BERT position embeddings 512→1024.

Первые 512 позиционных векторов сохранены без изменения, новые 512 инициализируются их копией и дообучаются. Обновлены position_ids, token_type_ids, config и tokenizer; это не просто изменение аргумента токенизатора. Инициализация не гарантирует качественное использование длинного контекста: нужны длинные обучающие примеры и отдельная оценка. Сложность обычного attention растёт квадратично с длиной окна.

7 тестов прошли, включая forward/backward на 1024 токенах, ненулевой градиент позиции 900 и offline save/reload. Локальный smoke с gradient checkpointing и неполной последней группой накопления тоже прошёл. GPU-память на реальном RuBERT локально не проверялась.

```bash
bash scripts/push_kaggle_rubert.sh --out /tmp/pii-rubert-1024-ready
# После подтверждённой остановки старой сессии:
bash scripts/push_kaggle_rubert.sh --submit
```

Следующий private kernel: `theodaimones888/ru-pii-recall-base-1024-v1`. Он отправлен; актуальный способ запуска описан ниже. Остановить старую сессию через доступный CLI не удалось: SDK CancelKernelSession требует session ID, которого status/get-kernel не возвращают. Kernel ID 135493280 не является session ID и для отмены не подставлялся. Пользователю отправлена просьба остановить старый запуск в интерфейсе Kaggle. [Ограничение API](https://github.com/Kaggle/kaggle-cli/issues/1169).


## Запуск 1024 с повторным использованием корпуса

По команде пользователя «запускай» отправлен kernel version 1:
https://www.kaggle.com/code/theodaimones888/ru-pii-recall-base-1024-v1

Задействован ранее загруженный приватный датасет `theodaimones888/ru-pii-recall-base-v1-bundle`. Повторная загрузка корпуса и создание нового датасета не выполнялись; переданы только текущий training code, schema/metrics и настройки. Старый config внутри датасета переопределяется встроенной конфигурацией нового runner; текущий код материализуется в `/kaggle/working/recall-code`.

```bash
bash scripts/push_kaggle_rubert.sh \
  --reuse-dataset theodaimones888/ru-pii-recall-base-v1-bundle --submit
.venv/bin/kaggle kernels status theodaimones888/ru-pii-recall-base-1024-v1
.venv/bin/kaggle kernels output theodaimones888/ru-pii-recall-base-1024-v1 -p artifacts/recall-rubert-1024
```

Статус старой сессии не подтверждён: её status endpoint вернул 403 Forbidden. Это не доказательство остановки. Отправка нового запуска выполнена по отдельной явной команде пользователя.
