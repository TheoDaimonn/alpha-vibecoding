#!/usr/bin/env python3
"""Build train/notebooks/distill_rubert_pii_onnx.ipynb.

The notebook distills models/rubert-base-pii-ner into a 4-layer (3x depth
reduction) student at near-teacher accuracy, then exports it to ONNX and
quantizes to INT8, measuring test quality before and after quantization.
Training data comes from data/csv. Regeneration overwrites the notebook;
manual edits are lost.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "train" / "notebooks" / "distill_rubert_pii_onnx.ipynb"

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", textwrap.dedent(src).strip() + "\n"))


def code(src: str) -> None:
    CELLS.append(("code", textwrap.dedent(src).strip() + "\n"))


md(
    """
    # Дистилляция rubert-base-pii-ner → 6-слойный студент + ONNX INT8

    Пайплайн сжатия детектора ПД для CPU-инференса по SLA хакатона:

    1. **Этап 1 — учитель:** `models/rubert-base-pii-ner` (12 слоёв, hidden 768)
       адаптируется к нашей таксономии: голова 43 классов заменяется на BIO-голову
       нашей схемы, дообучение до 5 эпох с ранней остановкой по val_loss (patience 2)
       на `data/csv/train.csv`; лучший чекпоинт сохраняется в `model_distilled/teacher_best/`.
    2. **Этап 2 — студент:** равномерно выбираем 4 из 12 слоёв учителя (0,3,6,9) —
       втрое меньше слоёв ⇒ ~3× меньше FLOPs/latency при том же hidden и словаре.
       Дообучаем с KD-loss: KL(учитель‖студент)/T + CE по жёстким меткам.
    3. **ONNX:** экспорт FP32 → динамическая INT8-квантизация.
    4. **Качество на test:** учитель / студент FP32 / студент ONNX FP32 / ONNX INT8 —
       span-F1 (точное совпадение start/end/label), char-coverage, per-label таблица
       **до и после квантизации**.
    5. **Латентность на CPU:** ONNX FP32 vs INT8.

    Ранняя остановка и выбор чекпоинта на обоих этапах обучения — по **val_loss на dev**
    (plain CE без class weights): если лосс не улучшался 2 эпохи подряд,
    обучение останавливается и берутся веса с минимальным val_loss.

    Маскирование/демаскирование — логика сервиса, здесь только детекция.

    ## Как запускать

    - GPU-ядро (CUDA), ячейки сверху вниз. Полный прогон: ~2–4 ч в зависимости от GPU.
    - Данные: `data/csv/{train,dev,test}.csv` (env `PII_DATA_DIR`); вывод в
      `model_distilled/` (env `PII_OUT_DIR`).
    - Smoke-режим: env `PII_MAX_ROWS=300 PII_TEACHER_NAME=<tiny-model>` — быстрый прогон
      всех ячеек без обучения всерьёз.
    - Этап 1 пропускается автоматически, если `model_distilled/teacher_best/` уже существует
      (перезапуск ноутбука → сразу этап 2). Переобучить учителя принудительно:
      env `PII_FORCE_TEACHER_TRAIN=1`.
    """
)

md("## Setup")

code(
    r"""
    import contextlib
    import copy
    import json
    import math
    import os
    import random
    import time
    from collections import Counter
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    import transformers
    from tqdm.auto import tqdm
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    SEED = 42
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"torch={torch.__version__} transformers={transformers.__version__} device={DEVICE}")
    """
)

md("## Конфигурация")

code(
    r"""
    def find_data_dir():
        env = os.environ.get("PII_DATA_DIR")
        if env:
            return Path(env)
        for cand in [Path.cwd(), *Path.cwd().parents]:
            for sub in ("data/csv", "csv"):
                if (cand / sub / "train.csv").exists():
                    return cand / sub
        return Path.cwd() / "data" / "csv"

    MAX_ROWS = int(os.environ.get("PII_MAX_ROWS", "0")) or None

    DATA_DIR = find_data_dir()
    OUT_DIR = Path(os.environ.get("PII_OUT_DIR", str(DATA_DIR.parent / "model_distilled")))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    TEACHER_NAME = os.environ.get("PII_TEACHER_NAME", "models/rubert-base-pii-ner")
    MAX_LEN = 512          # нативный предел BERT
    STRIDE = 128

    PII_LABELS = [
        "APARTMENT", "BIRTH_DATE", "BIRTH_PLACE", "CARDHOLDER", "CARD_NUMBER",
        "CITIZENSHIP", "CITY", "COUNTRY", "CVV", "DRIVER_LICENSE_NUMBER",
        "DRIVER_LICENSE_SERIES", "EMAIL", "HOUSE", "INN", "PASSPORT_DIVISION_CODE",
        "PASSPORT_ISSUER", "PASSPORT_ISSUE_DATE", "PASSPORT_NUMBER", "PASSPORT_SERIES",
        "PERSON", "PHONE", "PIN", "POSTAL_CODE", "STREET",
        # бонусные типы (документы кроме паспорта РФ, ТЗ п.6) — есть в train
        "SNILS", "OMS", "MILITARY_ID", "BIRTH_CERTIFICATE", "IP_ADDRESS",
    ]
    LABELS = PII_LABELS

    TAGS = ["O"] + [f"{p}-{lab}" for lab in LABELS for p in ("B", "I")]
    TAG2ID = {t: i for i, t in enumerate(TAGS)}
    ID2TAG = {i: t for t, i in TAG2ID.items()}
    N_CLASSES = len(TAGS)

    HP = {
        "teacher_epochs": int(os.environ.get("PII_TEACHER_EPOCHS", "5")),
        "distill_epochs": int(os.environ.get("PII_DISTILL_EPOCHS", "10")),
        "batch_size": int(os.environ.get("PII_BATCH_SIZE", "128")),
        "eval_batch_size": int(os.environ.get("PII_EVAL_BATCH_SIZE", "256")),
        "teacher_lr": 3e-5,
        "student_lr": 4e-5,
        "weight_decay": 0.01,
        "warmup_frac": 0.1,
        "grad_clip": 1.0,
        "weight_cap": 10.0,
        "temperature": 2.0,      # T для KD
        "alpha_hard": 0.5,      # вес CE (жёсткие метки); (1-alpha) — вес KL
        "student_layers": int(os.environ.get("PII_STUDENT_LAYERS", "4")),  # 12 -> 4: сжатие ~3x
        "patience": 2,         # эпох без улучшения val_loss до ранней остановки
        "dev_eval_rows": 4000,  # подвыборка dev для ежепоховой оценки
    }

    EXPECTED_COUNTS = {"train": 132087, "dev": 16000, "test": 16078}

    assert DATA_DIR.is_dir(), f"data dir not found: {DATA_DIR}"
    print(f"{len(LABELS)} labels, {N_CLASSES} BIO classes")
    print(f"data={DATA_DIR} | out={OUT_DIR} | teacher={TEACHER_NAME}")
    print(f"HP={HP}")
    """
)

md("## Данные")

code(
    r"""
    def load_split(name):
        df = pd.read_csv(DATA_DIR / f"{name}.csv")
        rows = [
            {"id": r.id, "source": r.source, "text": r.text, "entities": json.loads(r.entities)}
            for r in df.itertuples(index=False)
        ]
        if MAX_ROWS:
            rows = rows[:MAX_ROWS]
        if len(rows) != EXPECTED_COUNTS[name]:
            print(f"note: {name}: {len(rows)} rows (expected {EXPECTED_COUNTS[name]})")
        return rows

    train_rows = load_split("train")
    dev_rows = load_split("dev")
    test_rows = load_split("test")

    for name, rows in (("train", train_rows), ("dev", dev_rows), ("test", test_rows)):
        n_ent = sum(len(r["entities"]) for r in rows)
        print(f"{name}: {len(rows)} texts, {n_ent} entities")

    for name, rows in (("train", train_rows), ("dev", dev_rows), ("test", test_rows)):
        for r in random.sample(rows, min(200, len(rows))):
            for e in r["entities"]:
                assert r["text"][e["start"]:e["end"]] == e["text"], (name, r["id"], e)
    print("offset sanity: OK")

    used_labels = Counter(e["label"] for r in train_rows for e in r["entities"])
    # ADDRESS — контейнер, в BIO не кодируется (выводится из частей на уровне сервиса)
    unknown = set(used_labels) - set(LABELS) - {"ADDRESS"}
    assert not unknown, f"labels in data but not in schema: {unknown}"
    missing = set(LABELS) - set(used_labels)
    if missing:
        print(f"labels without train support (will stay at head, untrained): {sorted(missing)}")
    """
)

md("## BIO-теггинг")

code(
    r"""
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)
    print(type(tokenizer).__name__, "vocab:", tokenizer.vocab_size, "model_max_length:", tokenizer.model_max_length)
    assert MAX_LEN <= tokenizer.model_max_length

    PRIORITY = {"BIRTH_PLACE": 1}


    def span_priority(ent):
        return (PRIORITY.get(ent["label"], 2), ent["end"] - ent["start"], ent["start"], ent["label"])


    def token_labels_from_entities(entities, offsets):
        ents = [e for e in entities if e["label"] != "ADDRESS"]
        labels = []
        for s, e in offsets:
            if s == e:
                labels.append(-100)
                continue
            covering = [x for x in ents if x["start"] < e and x["end"] > s]
            if not covering:
                labels.append(0)
                continue
            best = min(covering, key=span_priority)
            prefix = "B" if s == best["start"] else "I"
            labels.append(TAG2ID[f"{prefix}-{best['label']}"])
        return labels


    def bio_decode(offsets, tag_ids):
        spans = []
        cur = None
        for (s, e), tid in zip(offsets, tag_ids):
            tag = ID2TAG[tid]
            if tag == "O":
                if cur is not None:
                    spans.append(cur)
                    cur = None
                continue
            prefix, label = tag.split("-", 1)
            if prefix == "B" or cur is None or cur[2] != label:
                if cur is not None:
                    spans.append(cur)
                cur = (s, e, label)
            else:
                cur = (cur[0], e, cur[2])
        if cur is not None:
            spans.append(cur)
        return spans


    def canonical_gold_spans(text, entities):
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False, truncation=False)
        labels = token_labels_from_entities(entities, enc["offset_mapping"])
        return bio_decode(enc["offset_mapping"], labels)
    """
)

md(
    """
    ## Сжатие словаря (опционально)

    Эмбеддинги (vocab ~120k x 768, ~92M параметров) после сжатия до 4 слоёв — главная
    часть модели. При `PII_PRUNE_VOCAB=1` словарь обрезается до токенов, реально
    встречающихся в train+dev (не test!), плюс специальные токены и все одиночные
    символы (страховка от опечаток и e-mail'ов). Строится remap old_id -> new_id;
    выброшенные токены идут в `[UNK]`. Учитель и студент обучаются/оцениваются на
    remapped-входах, поэтому сравнение честное. Сервис обязан применять remap после
    токенизации (артефакт `vocab_remap.npy`).
    """
)

code(
    r"""
    PRUNE_VOCAB = os.environ.get("PII_PRUNE_VOCAB", "0") == "1"
    PRUNE_MIN_COUNT = int(os.environ.get("PII_PRUNE_MIN_COUNT", "1"))

    PAD_ID = tokenizer.pad_token_id
    REMAP = None  # np.int64[old_vocab] -> new id; None = без обрезки


    def prune_model_embeddings(model):
        # обрезает word_embeddings до keep_sorted (порядок сохраняется: new_id = позиция)
        emb = model.bert.embeddings.word_embeddings
        idx = torch.from_numpy(keep_sorted).to(emb.weight.device)
        new_rows = emb.weight.data[idx].clone()
        model.resize_token_embeddings(len(keep_sorted))
        emb.weight.data = new_rows
        model.config.vocab_size = len(keep_sorted)
        return model


    if PRUNE_VOCAB:
        from tqdm.auto import trange

        stat_enc = tokenizer(
            [r["text"] for r in train_rows] + [r["text"] for r in dev_rows],
            truncation=True,
            max_length=MAX_LEN,
        )
        tok_counts = Counter()
        for ids in tqdm(stat_enc["input_ids"], desc="vocab stats", leave=False):
            tok_counts.update(ids)
        del stat_enc
        vocab = tokenizer.get_vocab()
        keep_ids = set(tokenizer.all_special_ids)
        for tok, tid in vocab.items():
            if len(tok) == 1 or tok_counts.get(tid, 0) >= PRUNE_MIN_COUNT:
                keep_ids.add(tid)
        keep_sorted = np.array(sorted(keep_ids), dtype=np.int64)
        REMAP = np.full(len(vocab), -1, dtype=np.int64)
        REMAP[keep_sorted] = np.arange(len(keep_sorted), dtype=np.int64)
        unk_new = int(REMAP[tokenizer.unk_token_id])  # unk — special, всегда в keep
        REMAP[REMAP < 0] = unk_new
        PAD_ID = int(REMAP[tokenizer.pad_token_id])
        print(f"vocab: {len(vocab)} -> {len(keep_sorted)} ({len(keep_sorted) / len(vocab):.1%}); "
              f"OOV-токенов в train+dev: {sum(c for tid, c in tok_counts.items() if tid not in keep_ids)}")
    else:
        print("vocab pruning disabled (PII_PRUNE_VOCAB=1 to enable)")


    def remap_ids(ids):
        if REMAP is None:
            return list(ids)
        return REMAP[np.array(ids, dtype=np.int64)].tolist()
    """
)

md("## Датасет и class weights")

code(
    r"""
    def build_samples(rows):
        enc = tokenizer(
            [r["text"] for r in rows],
            return_offsets_mapping=True,
            truncation=True,
            max_length=MAX_LEN,
            stride=STRIDE,
            return_overflowing_tokens=True,
        )
        owner = enc["overflow_to_sample_mapping"]
        samples = []
        for i in range(len(enc["input_ids"])):
            row = rows[owner[i]]
            samples.append({
                "input_ids": remap_ids(enc["input_ids"][i]),
                "labels": token_labels_from_entities(row["entities"], enc["offset_mapping"][i]),
                "n_tokens": len(enc["input_ids"][i]),
            })
        return samples


    t0 = time.time()
    train_samples = build_samples(train_rows)
    dev_samples = build_samples(dev_rows)
    test_samples = build_samples(test_rows)
    print(f"chunks: train={len(train_samples)} dev={len(dev_samples)} test={len(test_samples)} ({time.time() - t0:.1f}s)")

    tag_counts = Counter()
    for smp in train_samples:
        tag_counts.update(t for t in smp["labels"] if t != -100)
    counts = np.zeros(N_CLASSES)
    for tag_id, c in tag_counts.items():
        counts[tag_id] = c
    non_o = counts[1:]
    ref = float(np.median(non_o[non_o > 0]))
    weights = np.ones(N_CLASSES)
    for i in range(1, N_CLASSES):
        if counts[i] > 0:
            weights[i] = min(HP["weight_cap"], max(1.0, (ref / counts[i]) ** 0.5))
    class_weights = torch.tensor(weights, dtype=torch.float32)
    print("class weights: min={:.2f} max={:.2f}".format(weights[1:].min(), weights[1:].max()))
    """
)

md("## Инференс и метрики")

code(
    r"""
    def collate(samples):
        n = max(len(s["input_ids"]) for s in samples)
        input_ids, attention_mask, labels = [], [], []
        for s in samples:
            ids = s["input_ids"]
            pad = n - len(ids)
            input_ids.append(list(ids) + [PAD_ID] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
            lab = s.get("labels")
            if lab is None:
                labels.append([-100] * n)
            else:
                labels.append(list(lab) + [-100] * pad)
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


    def autocast_ctx():
        if DEVICE.type == "cuda":
            return torch.autocast("cuda", dtype=torch.float16)
        if DEVICE.type == "cpu":
            return torch.autocast("cpu", dtype=torch.bfloat16)
        return contextlib.nullcontext()


    @torch.no_grad()
    def eval_loss(model, samples, batch_size=None):
        # Val loss: plain CE без class weights — стабильный критерий ранней остановки.
        was_training = model.training
        model.eval()
        bs = batch_size or HP["eval_batch_size"]
        order = sorted(range(len(samples)), key=lambda i: samples[i]["n_tokens"])
        ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
        losses = []
        for i in range(0, len(order), bs):
            batch = collate([samples[j] for j in order[i:i + bs]])
            input_ids, attention_mask, labels = (t.to(DEVICE) for t in batch)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            losses.append(ce(logits.reshape(-1, N_CLASSES), labels.reshape(-1)).item())
        if was_training:
            model.train()
        return float(np.mean(losses))


    @torch.no_grad()
    def predict_texts(model, texts, batch_size=None):
        was_training = model.training
        model.eval()
        bs = batch_size or HP["eval_batch_size"]
        enc = tokenizer(
            texts,
            return_offsets_mapping=True,
            truncation=True,
            max_length=MAX_LEN,
            stride=STRIDE,
            return_overflowing_tokens=True,
        )
        owner = enc["overflow_to_sample_mapping"]
        votes = [dict() for _ in texts]
        order = sorted(range(len(enc["input_ids"])), key=lambda i: len(enc["input_ids"][i]))
        for i in range(0, len(order), bs):
            idxs = order[i:i + bs]
            batch = [{"input_ids": remap_ids(enc["input_ids"][j])} for j in idxs]
            input_ids, attention_mask, _ = collate(batch)
            logits = model(input_ids=input_ids.to(DEVICE), attention_mask=attention_mask.to(DEVICE)).logits
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            for bi, j in enumerate(idxs):
                text_idx = owner[j]
                for k, (s, e) in enumerate(enc["offset_mapping"][j]):
                    if s == e:
                        continue
                    key = (s, e)
                    p = float(conf[bi, k])
                    cur = votes[text_idx].get(key)
                    if cur is None or p > cur[1]:
                        votes[text_idx][key] = (int(pred[bi, k]), p)
        if was_training:
            model.train()
        return [bio_decode(sorted(v), [v[k][0] for k in sorted(v)]) for v in votes]


    def maskable_spans(spans):
        spans = sorted(spans, key=lambda x: (x[0], x[1]))
        out = []
        for s, e, lab in spans:
            if out and s < out[-1][1]:
                continue
            out.append((s, e, lab))
        return out


    def span_prf(gold_lists, pred_lists):
        tp, fp, fn = Counter(), Counter(), Counter()
        for gold, pred in zip(gold_lists, pred_lists):
            gs, ps = set(gold), set(pred)
            for sp in ps:
                if sp in gs:
                    tp[sp[2]] += 1
                else:
                    fp[sp[2]] += 1
            for sg in gs:
                if sg not in ps:
                    fn[sg[2]] += 1
        rows = []
        for lab in sorted(set(tp) | set(fp) | set(fn)):
            p = tp[lab] / (tp[lab] + fp[lab]) if tp[lab] + fp[lab] else 0.0
            r = tp[lab] / (tp[lab] + fn[lab]) if tp[lab] + fn[lab] else 0.0
            f = 2 * p * r / (p + r) if p + r else 0.0
            rows.append((lab, tp[lab], fp[lab], fn[lab], p, r, f))
        df = pd.DataFrame(rows, columns=["label", "tp", "fp", "fn", "precision", "recall", "f1"]).sort_values("f1")
        TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
        micro_p = TP / (TP + FP) if TP + FP else 0.0
        micro_r = TP / (TP + FN) if TP + FN else 0.0
        micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
        present = [row for row in rows if row[1] + row[3] > 0]
        macro_f = float(np.mean([row[6] for row in present])) if present else 0.0
        return micro_f, macro_f, df


    def _char_positions(spans):
        pos = set()
        for s, e, _ in maskable_spans(spans):
            pos.update(range(s, e))
        return pos


    def coverage_stats(gold_lists, pred_lists):
        inter = union_gold = union_pred = 0
        for gold, pred in zip(gold_lists, pred_lists):
            gp = _char_positions(gold)
            pp = _char_positions(pred)
            inter += len(gp & pp)
            union_gold += len(gp)
            union_pred += len(pp)
        p = inter / union_pred if union_pred else 1.0
        r = inter / union_gold if union_gold else 1.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f


    def text_char_f1(gold, pred):
        gp = _char_positions(gold)
        pp = _char_positions(pred)
        if not gp and not pp:
            return 1.0
        inter = len(gp & pp)
        p = inter / len(pp) if pp else 1.0
        r = inter / len(gp) if gp else 1.0
        return 2 * p * r / (p + r) if p + r else 0.0


    def render_spans(text, spans):
        out, cursor = [], 0
        for s, e, lab in sorted(spans):
            if s < cursor:
                continue
            out.append(text[cursor:s])
            out.append(f"[{lab}: {text[s:e]}]")
            cursor = e
        out.append(text[cursor:])
        return "".join(out)
    """
)

md("## Этап 1 — адаптация учителя к нашей таксономии")

code(
    r"""
    teacher_ckpt = OUT_DIR / ("teacher_best_pruned" if PRUNE_VOCAB else "teacher_best")
    full_ckpt = OUT_DIR / "teacher_best"
    # этап 1 пропускается, если подходящий чекпоинт уже есть (переобучение — PII_FORCE_TEACHER_TRAIN=1)
    force_teacher = os.environ.get("PII_FORCE_TEACHER_TRAIN", "0") == "1"
    have_ckpt = teacher_ckpt.exists() or (PRUNE_VOCAB and full_ckpt.exists())

    if teacher_ckpt.exists():
        teacher = AutoModelForTokenClassification.from_pretrained(teacher_ckpt).to(DEVICE)
        print("teacher loaded from", teacher_ckpt)
    elif PRUNE_VOCAB and full_ckpt.exists():
        teacher = AutoModelForTokenClassification.from_pretrained(full_ckpt).to(DEVICE)
        prune_model_embeddings(teacher)
        teacher.save_pretrained(teacher_ckpt)
        print("teacher pruned from full checkpoint ->", teacher_ckpt)
    else:
        teacher = AutoModelForTokenClassification.from_pretrained(
            TEACHER_NAME,
            num_labels=N_CLASSES,
            id2label={i: t for t, i in TAG2ID.items()},
            label2id=TAG2ID,
            ignore_mismatched_sizes=True,
        ).to(DEVICE)
        if PRUNE_VOCAB:
            prune_model_embeddings(teacher)
    assert teacher.config.model_type == "bert", f"unexpected arch: {teacher.config.model_type}"
    print(f"teacher layers={teacher.config.num_hidden_layers} hidden={teacher.config.hidden_size} vocab={teacher.config.vocab_size}")

    dev_texts = [r["text"] for r in dev_rows]
    gold_dev = [canonical_gold_spans(r["text"], r["entities"]) for r in dev_rows]
    eval_rng = random.Random(SEED)
    dev_eval_idx = eval_rng.sample(range(len(dev_rows)), min(HP["dev_eval_rows"], len(dev_rows)))
    dev_eval_texts = [dev_texts[i] for i in dev_eval_idx]
    dev_eval_gold = [gold_dev[i] for i in dev_eval_idx]


    def epoch_batches(samples, batch_size, rng):
        order = sorted(range(len(samples)), key=lambda i: samples[i]["n_tokens"] + rng.random() * 3)
        batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        rng.shuffle(batches)
        return batches


    if HP["teacher_epochs"] > 0 and (force_teacher or not have_ckpt):
        train_gen = random.Random(SEED)
        steps_per_epoch = math.ceil(len(train_samples) / HP["batch_size"])
        total_steps = steps_per_epoch * HP["teacher_epochs"]
        warmup_steps = int(HP["warmup_frac"] * total_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        optimizer = torch.optim.AdamW(teacher.parameters(), lr=HP["teacher_lr"], weight_decay=HP["weight_decay"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(DEVICE), ignore_index=-100)
        scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

        best_loss, best_epoch = float("inf"), 0
        for epoch in range(1, HP["teacher_epochs"] + 1):
            teacher.train()
            t0, losses = time.time(), []
            bar = tqdm(epoch_batches(train_samples, HP["batch_size"], train_gen), total=steps_per_epoch, desc=f"teacher epoch {epoch}", leave=False)
            for batch_idx in bar:
                batch = collate([train_samples[i] for i in batch_idx])
                input_ids, attention_mask, labels = (t.to(DEVICE) for t in batch)
                with autocast_ctx():
                    logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits
                    loss = loss_fn(logits.reshape(-1, N_CLASSES), labels.reshape(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), HP["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                losses.append(loss.item())
                bar.set_postfix(loss=f"{np.mean(losses[-50:]):.4f}")
            val_loss = eval_loss(teacher, dev_samples)
            pred_dev = predict_texts(teacher, dev_eval_texts)
            micro_f1, macro_f1, _ = span_prf(dev_eval_gold, pred_dev)
            marker = ""
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                teacher.save_pretrained(teacher_ckpt)
                marker = " <= best"
            print(f"teacher epoch {epoch}: loss={np.mean(losses):.4f} val_loss={val_loss:.4f} dev_micro={micro_f1:.4f} dev_macro={macro_f1:.4f}{marker} ({time.time() - t0:.0f}s)")
            if epoch - best_epoch >= HP["patience"]:
                print(f"early stop (teacher): val_loss not improving for {HP['patience']} epochs")
                break
    if teacher_ckpt.exists():
        teacher = AutoModelForTokenClassification.from_pretrained(teacher_ckpt).to(DEVICE)
        print("teacher loaded from", teacher_ckpt)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    if not teacher_ckpt.exists():
        print("warning: teacher untrained (no checkpoint; run stage 1 or set PII_TEACHER_EPOCHS>0)")
    """
)

md("## A/B: влияние обрезки словаря (если включена)")

code(
    r"""
    if PRUNE_VOCAB and full_ckpt.exists():
        teacher_full = AutoModelForTokenClassification.from_pretrained(full_ckpt).to(DEVICE)
        _, macro_full, _ = span_prf(dev_eval_gold, predict_texts(teacher_full, dev_eval_texts))
        _, macro_pruned, _ = span_prf(dev_eval_gold, predict_texts(teacher, dev_eval_texts))
        print(f"vocab pruning A/B (dev, {len(dev_eval_texts)} texts): "
              f"macro span-F1 full={macro_full:.4f} pruned={macro_pruned:.4f} delta={macro_pruned - macro_full:+.4f}")
        del teacher_full
    else:
        print("A/B skipped: pruning disabled or full-vocab checkpoint not found")
    """
)

md("## Этап 2 — студент: 4 из 12 слоёв учителя")

code(
    r"""
    # этап 2 всегда начинается с лучшего сохранённого учителя
    teacher = AutoModelForTokenClassification.from_pretrained(teacher_ckpt).to(DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student_cfg = copy.deepcopy(teacher.config)
    L = teacher.config.num_hidden_layers
    S = min(HP["student_layers"], L)
    sel = sorted({min(L - 1, round(i * L / S)) for i in range(S)})
    assert len(sel) >= 1
    student_cfg.num_hidden_layers = len(sel)
    student = AutoModelForTokenClassification.from_config(student_cfg)
    student.bert.embeddings.load_state_dict(teacher.bert.embeddings.state_dict())
    for i, src_idx in enumerate(sel):
        student.bert.encoder.layer[i].load_state_dict(teacher.bert.encoder.layer[src_idx].state_dict())
    student.classifier.load_state_dict(teacher.classifier.state_dict())
    student.to(DEVICE)

    def n_params(m):
        return sum(p.numel() for p in m.parameters())

    t_params, s_params = n_params(teacher), n_params(student)
    print(f"teacher: {t_params / 1e6:.1f}M params, {L} layers")
    print(f"student: {s_params / 1e6:.1f}M params, {len(sel)} layers (teacher layers {sel})")
    print(f"compute compression (layers): {L / len(sel):.1f}x")
    print(f"params compression: {t_params / s_params:.2f}x (vocab-эмбеддинги остаются общими)")
    """
)

md("## Этап 2 — дистилляция (KL + жёсткие метки)")

code(
    r"""
    train_gen = random.Random(SEED)
    steps_per_epoch = math.ceil(len(train_samples) / HP["batch_size"])
    total_steps = steps_per_epoch * HP["distill_epochs"]
    warmup_steps = int(HP["warmup_frac"] * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    optimizer = torch.optim.AdamW(student.parameters(), lr=HP["student_lr"], weight_decay=HP["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ce_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(DEVICE), ignore_index=-100)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    T = HP["temperature"]
    alpha = HP["alpha_hard"]

    student_ckpt = OUT_DIR / "student_best"
    best_loss, best_epoch = float("inf"), 0
    history = []
    t_start = time.time()
    for epoch in range(1, HP["distill_epochs"] + 1):
        student.train()
        t0, losses = time.time(), []
        bar = tqdm(epoch_batches(train_samples, HP["batch_size"], train_gen), total=steps_per_epoch, desc=f"distill epoch {epoch}", leave=False)
        for batch_idx in bar:
            batch = collate([train_samples[i] for i in batch_idx])
            input_ids, attention_mask, labels = (t.to(DEVICE) for t in batch)
            with torch.no_grad(), autocast_ctx():
                t_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits
            with autocast_ctx():
                s_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits
                ce = ce_fn(s_logits.reshape(-1, N_CLASSES), labels.reshape(-1))
                log_p = torch.nn.functional.log_softmax(s_logits.float() / T, dim=-1)
                q = torch.nn.functional.softmax(t_logits.float() / T, dim=-1)
                kl_tok = torch.nn.functional.kl_div(log_p, q, reduction="none").sum(-1)
                mask = labels != -100
                kl = (kl_tok * mask).sum() / mask.sum().clamp(min=1)
                loss = alpha * ce + (1.0 - alpha) * (T ** 2) * kl
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), HP["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            bar.set_postfix(loss=f"{np.mean(losses[-50:]):.4f}")
        val_loss = eval_loss(student, dev_samples)
        pred_dev = predict_texts(student, dev_eval_texts)
        micro_f1, macro_f1, _ = span_prf(dev_eval_gold, pred_dev)
        marker = ""
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            student.save_pretrained(student_ckpt)
            marker = " <= best"
        history.append({"epoch": epoch, "loss": round(float(np.mean(losses)), 4),
                        "val_loss": round(val_loss, 4),
                        "dev_micro_f1": round(micro_f1, 4), "dev_macro_f1": round(macro_f1, 4),
                        "sec": round(time.time() - t0, 1)})
        print(f"distill epoch {epoch}: loss={np.mean(losses):.4f} val_loss={val_loss:.4f} dev_micro={micro_f1:.4f} dev_macro={macro_f1:.4f}{marker}")
        if epoch - best_epoch >= HP["patience"]:
            print(f"early stop (student): val_loss not improving for {HP['patience']} epochs")
            break

    print(f"total {time.time() - t_start:.0f}s, best epoch {best_epoch} (val_loss={best_loss:.4f})")
    student = AutoModelForTokenClassification.from_pretrained(student_ckpt).to(DEVICE)
    student.eval()
    print(pd.DataFrame(history).to_string(index=False))
    """
)

md("## Качество на test — до квантизации")

code(
    r"""
    test_texts = [r["text"] for r in test_rows]
    gold_test = [canonical_gold_spans(r["text"], r["entities"]) for r in test_rows]

    results = {}

    pred_teacher = predict_texts(teacher, test_texts)
    results["teacher (torch fp32)"] = span_prf(gold_test, pred_teacher)

    pred_student = predict_texts(student, test_texts)
    results["student (torch fp32)"] = span_prf(gold_test, pred_student)

    summary = []
    for name, preds in (("teacher (torch fp32)", pred_teacher), ("student (torch fp32)", pred_student)):
        micro_f1, macro_f1, _ = results[name]
        cov_p, cov_r, cov_f = coverage_stats(gold_test, preds)
        summary.append({"model": name, "span_micro_f1": round(micro_f1, 4),
                        "span_macro_f1": round(macro_f1, 4), "char_cov_f1": round(cov_f, 4)})
    print(pd.DataFrame(summary).to_string(index=False))

    _, _, per_label_student = results["student (torch fp32)"]
    print("\nPer-label span-F1 студента (худшие сверху):")
    print(per_label_student.to_string(index=False))
    """
)

md("## ONNX: экспорт FP32 и квантизация INT8")

code(
    r"""
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    student.eval()
    dummy = tokenizer("тестовая строка для экспорта onnx", return_tensors="pt")
    dummy_ids = torch.tensor([remap_ids(dummy["input_ids"][0])], dtype=torch.long).to(DEVICE)
    dummy_mask = dummy["attention_mask"].to(DEVICE)
    export_kwargs = dict(
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=17,
    )
    onnx_fp32 = OUT_DIR / "student.onnx"
    onnx_int8 = OUT_DIR / "student_int8.onnx"
    try:
        torch.onnx.export(student, (dummy_ids, dummy_mask), str(onnx_fp32), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(student, (dummy_ids, dummy_mask), str(onnx_fp32), **export_kwargs)
    quantize_dynamic(str(onnx_fp32), str(onnx_int8), weight_type=QuantType.QInt8)

    for path in (onnx_fp32, onnx_int8):
        print(f"{path.name}: {path.stat().st_size / 1e6:.1f} MB")

    with torch.no_grad():
        logits_pt = student(dummy_ids, dummy_mask).logits.cpu().numpy()
    sess_tmp = ort.InferenceSession(str(onnx_int8), providers=["CPUExecutionProvider"])
    logits_q = sess_tmp.run(None, {"input_ids": dummy_ids.cpu().numpy(), "attention_mask": dummy_mask.cpu().numpy()})[0]
    print(f"int8 argmax agreement vs torch: {(logits_q.argmax(-1) == logits_pt.argmax(-1)).mean():.4f}")
    """
)

md("## ONNX-инференс")

code(
    r"""
    def predict_texts_onnx(session, texts, batch_size=None):
        bs = batch_size or HP["eval_batch_size"]
        enc = tokenizer(
            texts,
            return_offsets_mapping=True,
            truncation=True,
            max_length=MAX_LEN,
            stride=STRIDE,
            return_overflowing_tokens=True,
        )
        owner = enc["overflow_to_sample_mapping"]
        votes = [dict() for _ in texts]
        order = sorted(range(len(enc["input_ids"])), key=lambda i: len(enc["input_ids"][i]))
        for i in range(0, len(order), bs):
            idxs = order[i:i + bs]
            batch = [{"input_ids": remap_ids(enc["input_ids"][j])} for j in idxs]
            input_ids, attention_mask, _ = collate(batch)
            logits = session.run(None, {
                "input_ids": input_ids.numpy(),
                "attention_mask": attention_mask.numpy(),
            })[0]
            logits = logits - logits.max(axis=-1, keepdims=True)
            e = np.exp(logits)
            probs = e / e.sum(axis=-1, keepdims=True)
            conf = probs.max(axis=-1)
            pred = probs.argmax(axis=-1)
            for bi, j in enumerate(idxs):
                text_idx = owner[j]
                for k, (s, e_off) in enumerate(enc["offset_mapping"][j]):
                    if s == e_off:
                        continue
                    key = (s, e_off)
                    p = float(conf[bi, k])
                    cur = votes[text_idx].get(key)
                    if cur is None or p > cur[1]:
                        votes[text_idx][key] = (int(pred[bi, k]), p)
        return [bio_decode(sorted(v), [v[k][0] for k in sorted(v)]) for v in votes]
    """
)

md("## Качество на test — после квантизации")

code(
    r"""
    sess_fp32 = ort.InferenceSession(str(onnx_fp32), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(onnx_int8), providers=["CPUExecutionProvider"])

    pred_onnx_fp32 = predict_texts_onnx(sess_fp32, test_texts)
    results["student onnx fp32"] = span_prf(gold_test, pred_onnx_fp32)
    pred_onnx_int8 = predict_texts_onnx(sess_int8, test_texts)
    results["student onnx int8"] = span_prf(gold_test, pred_onnx_int8)

    pred_by_name = {
        "teacher (torch fp32)": pred_teacher,
        "student (torch fp32)": pred_student,
        "student onnx fp32": pred_onnx_fp32,
        "student onnx int8": pred_onnx_int8,
    }
    summary = []
    for name, preds in pred_by_name.items():
        micro_f1, macro_f1, _ = results[name]
        cov_p, cov_r, cov_f = coverage_stats(gold_test, preds)
        summary.append({"model": name, "span_micro_f1": round(micro_f1, 4),
                        "span_macro_f1": round(macro_f1, 4), "char_cov_f1": round(cov_f, 4)})
    print(pd.DataFrame(summary).to_string(index=False))

    _, _, per_int8 = results["student onnx int8"]
    _, _, per_fp32 = results["student (torch fp32)"]
    delta = per_int8.set_index("label")["f1"].sub(per_fp32.set_index("label")["f1"], fill_value=0.0)
    print("\nPer-label span-F1: изменение после INT8 (только ненулевые дельты):")
    print(delta[delta.abs() > 1e-9].sort_values().to_string() or "изменений нет")
    """
)

md("## Латентность на CPU: FP32 vs INT8")

code(
    r"""
    bench_rng = random.Random(SEED)
    bench_texts = [t for t in dev_texts if 80 <= len(t) <= 400][:256]
    enc = tokenizer(bench_texts, return_offsets_mapping=True, truncation=True, max_length=MAX_LEN,
                    stride=STRIDE, return_overflowing_tokens=True)
    chunks = [remap_ids(enc["input_ids"][i]) for i in range(len(enc["input_ids"]))]
    order = sorted(range(len(chunks)), key=lambda i: len(chunks[i]))
    bench_batches = []
    for i in range(0, len(order), HP["eval_batch_size"]):
        idxs = order[i:i + HP["eval_batch_size"]]
        input_ids, attention_mask, _ = collate([{"input_ids": chunks[j]} for j in idxs])
        bench_batches.append((input_ids.numpy(), attention_mask.numpy()))

    def bench(session):
        for ids, mask in bench_batches[:4]:
            session.run(None, {"input_ids": ids, "attention_mask": mask})
        t0 = time.time()
        n = 0
        for ids, mask in bench_batches:
            session.run(None, {"input_ids": ids, "attention_mask": mask})
            n += len(ids)
        dt = time.time() - t0
        return dt, n

    lat = []
    for name, path in (("onnx fp32", onnx_fp32), ("onnx int8", onnx_int8)):
        s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        dt, n = bench(s)
        lat.append({"model": name, "sec_total": round(dt, 2), "texts": n,
                    "ms_per_text": round(dt / n * 1000, 2), "texts_per_sec": round(n / dt, 1)})
    lat_df = pd.DataFrame(lat)
    print(lat_df.to_string(index=False))
    speedup = lat_df.loc[lat_df["model"] == "onnx fp32", "ms_per_text"].iloc[0] / lat_df.loc[lat_df["model"] == "onnx int8", "ms_per_text"].iloc[0]
    print(f"\nint8 speedup vs fp32: {speedup:.2f}x (CPU, batch={HP['eval_batch_size']})")
    """
)

md("## Артефакты")

code(
    r"""
    student.save_pretrained(OUT_DIR / "student_final")
    tokenizer.save_pretrained(OUT_DIR / "student_final")
    model_config = {
        "teacher_name": TEACHER_NAME,
        "student_layers": HP["student_layers"],
        "labels": LABELS,
        "tags": TAGS,
        "max_len": MAX_LEN,
        "stride": STRIDE,
        "hp": {k: v for k, v in HP.items()},
        "vocab_pruned": bool(PRUNE_VOCAB),
        "vocab_size": int(student.config.vocab_size),
        "files": {
            "onnx_fp32": onnx_fp32.name,
            "onnx_int8": onnx_int8.name,
        },
    }
    if PRUNE_VOCAB:
        np.save(OUT_DIR / "vocab_remap.npy", REMAP)
        model_config["vocab_remap"] = "vocab_remap.npy (old_tokenizer_id -> model_id; service: tokenizer -> remap -> model)"
    (OUT_DIR / "model_config.json").write_text(json.dumps(model_config, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved to", OUT_DIR)
    print("  student_final/  — HF-формат (torch)")
    print("  student.onnx    — FP32 для CPU-инференса")
    print("  student_int8.onnx — INT8, целевой артефакт для сервиса")

    demo_texts = [
        "Клиент Иванов Иван Иванович, паспорт 4509 123456, выдан ОВД Ленинского района г. Москвы, код подразделения 770-001.",
        "Оплата картой 4276 5500 1234 5678, держатель IVAN IVANOV, cvv 123, пин-код 9876.",
        "Как писал Пушкин, пишите на почту ivanova.o@mail.ru или звоните +7 916 123-45-67.",
    ]
    for demo_text in demo_texts:
        spans = predict_texts_onnx(sess_int8, [demo_text])[0]
        print("TEXT :", demo_text)
        print("SPANS:", [(demo_text[s:e], lab) for s, e, lab in spans])
        print()
    """
)


def main() -> int:
    for i, (kind, src) in enumerate(CELLS):
        if kind == "code":
            lines = [ln for ln in src.splitlines() if not ln.lstrip().startswith(("%", "!"))]
            compile("\n".join(lines), f"<cell {i}>", "exec")
    cells = []
    for i, (kind, src) in enumerate(CELLS):
        source = src.splitlines(keepends=True)
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "id": f"cell-{i}", "metadata": {}, "source": source})
        else:
            cells.append({
                "cell_type": "code",
                "id": f"cell-{i}",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": source,
            })
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    n_code = sum(1 for kind, _ in CELLS if kind == "code")
    print(f"written {OUT} ({len(CELLS)} cells: {n_code} code), compile check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
