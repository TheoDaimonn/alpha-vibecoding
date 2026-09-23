#!/usr/bin/env python3
"""Generate a synthetic supplement for under-represented PII labels.

The existing task-synthetic-supplement-v2 covers PIN, CARDHOLDER, CVV,
CITIZENSHIP, APARTMENT and the passport issuer fields from a single template
family, so a model trained on it may not generalize to the evaluation
phrasing. This script generates additional rows for those labels with varied
separators ("серия хххх номер хххххх", "№", dashes), date formats
(mm/dd/yyyy, dd.mm.yyyy, yyyy-mm-dd, spelled-out, two-digit years), registers
(keep/lower/upper) and surrounding contexts.

Labels follow the canonical schema; entities carry char offsets into the row
text; rows are validated (offset equality, no crossing spans) and appended to
data/csv with deterministic md5-based 80/10/10 split assignment, matching
scripts/integrate_new_pii.py conventions. Idempotent: existing rows of this
source are replaced on re-run. Stdlib only.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from hashlib import md5
from pathlib import Path
from random import Random

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = REPO_ROOT / "data" / "csv"
SPLITS = ("train", "dev", "test")
CSV_FIELDS = ("id", "split", "source", "text", "entities")
SOURCE = "task-synthetic-rare-v1"
SEED = 20260923
ROW_COUNT = 3000
ALLOWED_LABELS = frozenset(
    {
        "APARTMENT", "BIRTH_DATE", "CARDHOLDER", "CARD_NUMBER", "CITIZENSHIP",
        "CITY", "COUNTRY", "CVV", "HOUSE", "PASSPORT_DIVISION_CODE",
        "PASSPORT_ISSUE_DATE", "PASSPORT_ISSUER", "PASSPORT_NUMBER",
        "PASSPORT_SERIES", "PERSON", "PHONE", "POSTAL_CODE", "STREET", "PIN",
    }
)

SURNAMES_M = ["Соколов", "Смирнов", "Кузнецов", "Попов", "Волков", "Лебедев", "Козлов", "Новиков", "Морозов", "Петров", "Зайцев", "Соловьёв", "Васильев", "Голубев", "Богданов", "Воробьёв", "Фёдоров", "Михайлов", "Беляев", "Тарасов"]
SURNAMES_F = ["Соколова", "Смирнова", "Кузнецова", "Попова", "Волкова", "Лебедева", "Козлова", "Новикова", "Морозова", "Петрова", "Зайцева", "Соловьёва", "Васильева", "Голубева", "Богданова", "Воробьёва", "Фёдорова", "Михайлова", "Беляева", "Тарасова"]
MALE_NAMES = ["Иван", "Пётр", "Андрей", "Сергей", "Николай", "Максим", "Артём", "Дмитрий", "Роман", "Владимир"]
FEMALE_NAMES = ["Анна", "Мария", "Елена", "Ольга", "Наталья", "Ирина", "Татьяна", "Екатерина", "Светлана", "Юлия"]
PATRONYMICS_M = ["Иванович", "Петрович", "Андреевич", "Сергеевич", "Николаевич", "Максимович", "Артёмович", "Дмитриевич"]
PATRONYMICS_F = ["Ивановна", "Петровна", "Андреевна", "Сергеевна", "Николаевна", "Максимовна", "Дмитриевна"]
CITIES = ["Тула", "Рязань", "Псков", "Курск", "Тверь", "Уфа", "Пермь", "Омск", "Тюмень", "Вологда", "Смоленск", "Брянск", "Липецк", "Тамбов", "Калуга"]
REGIONS = ["Тульской", "Рязанской", "Псковской", "Курской", "Тверской", "Вологодской", "Смоленской", "Брянской", "Липецкой", "Тамбовской"]
DISTRICTS = ["Заречный", "Привокзальный", "Центральный", "Северный", "Октябрьский", "Ленинский", "Кировский", "Советский"]
STREETS = ["Ленина", "Мира", "Гагарина", "Садовая", "Полевая", "Луговая", "Центральная", "Молодёжная", "Парковая", "Сосновая", "Кленовая", "Берёзовая"]
ISSUER_TEMPLATES = [
    "ОУФМС России по {region} области",
    "УМВД России по г. {city}",
    "Отделом УФМС России по {region} области",
    "МФЦ {district} района г. {city}",
    "ГУ МВД России по {region} области",
    "ТП №{code} ОУФМС России по {region} области",
]
OPENERS = [
    "Проверьте, пожалуйста, данные клиента.",
    "Для оформления заявки необходимы следующие сведения:",
    "Данные из анкеты клиента:",
    "Обработайте входящее обращение:",
    "Сведения для проверки:",
    "Прошу подтвердить актуальность данных:",
    "",
    "",
    "",
]
CLOSERS = [
    "Заранее спасибо.",
    "С уважением, отдел контроля данных.",
    "Данные переданы на проверку.",
    "Обратите внимание на корректность заполнения.",
    "",
    "",
    "",
    "",
]


class Builder:
    """Assembles row text while tracking entity char offsets."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.entities: list[dict] = []
        self.pos = 0

    def add(self, text: str, label: str | None = None) -> None:
        if label is not None:
            self.entities.append(
                {"start": self.pos, "end": self.pos + len(text), "label": label, "text": text}
            )
        self.parts.append(text)
        self.pos += len(text)

    def space(self, sep: str = " ") -> None:
        if self.parts and sep:
            self.parts.append(sep)
            self.pos += len(sep)

    @property
    def text(self) -> str:
        return "".join(self.parts)


def digits(rng: Random, n: int) -> str:
    return "".join(rng.choice("0123456789") for _ in range(n))


def passport_series(rng: Random) -> str:
    return digits(rng, 4)


def full_name(rng: Random) -> str:
    if rng.random() < 0.5:
        return f"{rng.choice(SURNAMES_M)} {rng.choice(MALE_NAMES)} {rng.choice(PATRONYMICS_M)}"
    return f"{rng.choice(SURNAMES_F)} {rng.choice(FEMALE_NAMES)} {rng.choice(PATRONYMICS_F)}"


def card_number(rng: Random) -> str:
    raw = digits(rng, 16)
    form = rng.choice(["groups", "dashes", "plain"])
    if form == "groups":
        return " ".join(raw[i : i + 4] for i in range(0, 16, 4))
    if form == "dashes":
        return "-".join(raw[i : i + 4] for i in range(0, 16, 4))
    return raw


def cvv(rng: Random) -> str:
    return digits(rng, 3)


def pin(rng: Random) -> str:
    return digits(rng, 4)


def format_date(rng: Random) -> str:
    day = rng.randint(1, 28)
    month = rng.randint(1, 12)
    year = rng.randint(1990, 2024)
    months_gen = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"]
    form = rng.choice(["dmy", "ymd", "mdy", "dashed", "text", "short"])
    if form == "dmy":
        return f"{day:02d}.{month:02d}.{year}"
    if form == "ymd":
        return f"{year}-{month:02d}-{day:02d}"
    if form == "mdy":
        return f"{month:02d}/{day:02d}/{year}"
    if form == "dashed":
        return f"{year}.{day:02d}.{month:02d}"
    if form == "short":
        return f"{day:02d}.{month:02d}.{year % 100:02d}"
    return f"{day} {months_gen[month - 1]} {year} года"


def block_passport(rng: Random, b: Builder) -> None:
    series, number = passport_series(rng), digits(rng, 6)
    sep = rng.choice([" ", "-", ""])
    series_shown = f"{series[:2]}{sep}{series[2:]}" if sep else series
    head, middle = rng.choice(
        [
            ("паспорт РФ, серия ", " номер "),
            ("паспорт ", " № "),
            ("документ: серия ", ", номер "),
            ("серия ", ", рег. номер "),
            ("паспортные данные: серия ", " номер "),
        ]
    )
    b.add(head)
    b.add(series_shown, "PASSPORT_SERIES")
    b.add(middle)
    b.add(number, "PASSPORT_NUMBER")
    b.space("; ")
    issuer = rng.choice(ISSUER_TEMPLATES).format(
        region=rng.choice(REGIONS), city=rng.choice(CITIES),
        district=rng.choice(DISTRICTS), code=digits(rng, 2),
    )
    code = digits(rng, 3) + rng.choice([" ", "-", "/"]) + digits(rng, 3)
    date = format_date(rng)
    b.add("выдан " if rng.random() < 0.5 else "орган выдачи: ")
    b.add(issuer, "PASSPORT_ISSUER")
    b.space("; ")
    b.add(rng.choice(["код подразделения ", "к/п ", "код подр. ", "подразделение "]))
    b.add(code, "PASSPORT_DIVISION_CODE")
    b.space("; ")
    b.add(rng.choice(["дата выдачи ", "выдан ", "дата выдачи паспорта "]))
    b.add(date, "PASSPORT_ISSUE_DATE")


def block_card(rng: Random, b: Builder) -> None:
    b.add(rng.choice(["карта № ", "номер карты ", "банковская карта ", "платёжная карта № "]))
    b.add(card_number(rng), "CARD_NUMBER")
    b.space("; ")
    name = full_name(rng)
    if rng.random() < 0.4:
        holder = " ".join(w.upper() for w in name.split())
        b.add(rng.choice(["имя держателя: ", "владелец карты: ", "CARDHOLDER: ", "держатель: "]))
    else:
        holder = name
        b.add(rng.choice(["владелец карты ", "имя держателя карты ", "держатель карты "]))
    b.add(holder, "CARDHOLDER")
    b.space("; ")
    b.add(rng.choice(["CVV ", "CVV-код ", "cvv2 ", "код безопасности ", "CVC "]))
    b.add(cvv(rng), "CVV")
    b.space("; ")
    b.add(rng.choice(["пин-код ", "PIN ", "пин ", "PIN-код "]))
    b.add(pin(rng), "PIN")


def block_citizenship(rng: Random, b: Builder) -> None:
    value = rng.choice(
        ["Российская Федерация", "РФ", "Россия", "Российской Федерации", "России"]
    )
    phrase = rng.choice(
        [
            "гражданство: ",
            "гражданство клиента: ",
            "является гражданином ",
            "гражданка ",
            "гражданин ",
        ]
    )
    b.add(phrase)
    b.add(value, "CITIZENSHIP")


def block_address(rng: Random, b: Builder) -> None:
    if rng.random() < 0.5:
        b.add(rng.choice(["адрес регистрации: ", "адрес: ", "зарегистрирован по адресу: ", "проживает: "]))
    if rng.random() < 0.3:
        b.add(rng.choice(["Российская Федерация", "Россия"]), "COUNTRY")
        b.space(", ")
    if rng.random() < 0.3:
        b.add(rng.choice(["индекс ", "почтовый индекс "]))
        b.add(digits(rng, 6), "POSTAL_CODE")
        b.space(", ")
    b.add(rng.choice(["г. ", "город ", ""]))
    b.add(rng.choice(CITIES), "CITY")
    b.space(", ")
    b.add(rng.choice(["ул. ", "улица ", ""]))
    b.add(rng.choice(STREETS), "STREET")
    b.space(", ")
    b.add(rng.choice(["д. ", "дом ", ""]))
    b.add(f"{rng.randint(1, 199)}", "HOUSE")
    if rng.random() < 0.5:
        b.space(", ")
        b.add(rng.choice(["кв. ", "квартира ", "кв.№ ", "кв "]))
        b.add(f"{rng.randint(1, 489)}", "APARTMENT")


def block_person(rng: Random, b: Builder) -> None:
    b.add(rng.choice(["клиент ", "заявитель ", "ФИО: ", "обратился ", ""]))
    b.add(full_name(rng), "PERSON")
    if rng.random() < 0.4:
        b.space(", ")
        b.add("дата рождения ")
        b.add(format_date(rng), "BIRTH_DATE")


def apply_register(rng: Random, text: str) -> str:
    form = rng.random()
    if form < 0.25:
        return text.lower()
    if form < 0.4:
        return text.upper()
    return text


def validate_row(text: str, entities: list[dict], location: str) -> None:
    open_ends: list[int] = []
    for e in sorted(entities, key=lambda x: (x["start"], x["start"] - x["end"])):
        if text[e["start"]:e["end"]] != e["text"]:
            raise ValueError(f"{location}: offset mismatch {e!r}")
        if e["label"] not in ALLOWED_LABELS:
            raise ValueError(f"{location}: label {e['label']!r} not allowed")
        while open_ends and open_ends[-1] <= e["start"]:
            open_ends.pop()
        if open_ends and e["end"] > open_ends[-1]:
            raise ValueError(f"{location}: crossing spans at {e['text']!r}")
        open_ends.append(e["end"])


def split_for(key: str) -> str:
    bucket = int(md5(key.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "dev"
    return "test"


def generate_row(rng: Random, index: int) -> dict:
    b = Builder()
    opener = rng.choice(OPENERS)
    if opener:
        b.add(opener)
        b.space(" ")
    kind = rng.random()
    if kind < 0.35:
        if rng.random() < 0.5:
            block_person(rng, b)
            b.space(" ")
        block_passport(rng, b)
    elif kind < 0.6:
        block_card(rng, b)
    elif kind < 0.8:
        if rng.random() < 0.3:
            block_person(rng, b)
            b.space(" ")
        block_address(rng, b)
    elif kind < 0.9:
        block_citizenship(rng, b)
    else:
        block_person(rng, b)
        b.space(" ")
        block_passport(rng, b)
        b.space(" ")
        block_address(rng, b)
    closer = rng.choice(CLOSERS)
    if closer:
        b.space(" ")
        b.add(closer)
    text = apply_register(rng, b.text.strip())
    # recompute entity texts after register transform and strip
    shift = len(b.text) - len(b.text.lstrip())
    entities = []
    for e in b.entities:
        start = max(0, e["start"] - shift)
        end = e["end"] - shift
        entities.append({"start": start, "end": end, "label": e["label"], "text": text[start:end]})
    entities = [e for e in entities if e["end"] > e["start"]]
    entities.sort(key=lambda e: (e["start"], e["start"] - e["end"]))
    validate_row(text, entities, f"row[{index}]")
    key = " ".join(text.split()).strip().lower()
    return {"text": text, "entities": entities, "key": key}


def main() -> int:
    rng = Random(SEED)
    rows: list[dict] = []
    seen_keys: set[str] = set()
    index = 0
    while len(rows) < ROW_COUNT:
        row = generate_row(rng, index)
        index += 1
        if row["key"] in seen_keys:
            continue
        seen_keys.add(row["key"])
        rows.append(row)

    label_counts: Counter = Counter()
    additions: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for row in rows:
        split = split_for(row["key"])
        prefix = f"ru-task-rare-{split}"
        additions[split].append(
            {
                "id": f"{prefix}-{len(additions[split]):06d}",
                "split": split,
                "source": SOURCE,
                "text": row["text"],
                "entities": row["entities"],
            }
        )
        label_counts.update(e["label"] for e in row["entities"])

    for split in SPLITS:
        path = CSV_DIR / f"{split}.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            existing = [r for r in csv.DictReader(handle) if r["source"] != SOURCE]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for r in existing + additions[split]:
                out = dict(r)
                if isinstance(out["entities"], list):
                    out["entities"] = json.dumps(out["entities"], ensure_ascii=False, separators=(",", ":"))
                writer.writerow(out)
        print(f"{split}: +{len(additions[split])} rare-label rows (total {len(existing) + len(additions[split])})")
    print("label counts:", dict(sorted(label_counts.items(), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
