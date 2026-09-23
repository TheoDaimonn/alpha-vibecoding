"""Deterministic Russian synthetic supplement for task classes missing in public corpora.

This module is deliberately a SUPPLEMENT, not the primary corpus.  It generates
format-valid but non-issued values and exact character spans.  The hybrid build
requires source corpora as well, so training cannot silently become synthetic-only.
"""
from __future__ import annotations

import argparse
import json
import random
import string
from collections import Counter
from pathlib import Path
from typing import Any

from .schema import LABELS, TASK_REQUIRED_LABELS, write_jsonl, validate_record

SURNAME = {
    "train": "Иванов Петров Соколов Волков Орлов Морозов Павлов Козлов Новиков Фомин Беляев Жуков Макаров Захаров Осипов Егоров Никитин Данилов Киселёв Титов".split(),
    "dev": "Селиванов Ширяев Кондратьев Трофимов Быков Воронин Елисеев Федотов".split(),
    "test": "Лазарев Кудрявцев Родионов Калинин Куликов Мартынов Афанасьев Гаврилов".split(),
}
FIRST_M = "Иван Пётр Павел Олег Сергей Алексей Денис Роман Артём Илья Николай Андрей Максим Виктор Юрий Тимур".split()
FIRST_F = "Анна Елена Ольга Мария Ирина Наталья Татьяна Светлана Полина Вера".split()
PATR_M = "Иванович Петрович Сергеевич Андреевич Павлович Олегович Викторович Николаевич Алексеевич Юрьевич".split()
PATR_F = "Ивановна Петровна Сергеевна Андреевна Павловна Олеговна Викторовна Николаевна Алексеевна Юрьевна".split()
CITIES = ["Тула", "Омск", "Пермь", "Казань", "Самара", "Калуга", "Томск", "Курск", "Псков", "Уфа", "Сочи", "Вологда"]
STREETS = ["Мира", "Садовая", "Лесная", "Полевая", "Новая", "Набережная", "Сосновая", "Школьная", "Речная", "Тихая"]
REGIONS = ["Тульская область", "Самарская область", "Пермский край", "Омская область", "Республика Татарстан"]
CITIZENSHIPS = ["Российская Федерация", "Россия", "Республика Беларусь", "Казахстан"]
ISSUERS = ["УМВД России по Тульской области", "МВД по Республике Татарстан", "ОВД Центрального района", "УМВД России по Омской области"]
MONTHS = "января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split()

SUPPLEMENT_LABELS = frozenset(TASK_REQUIRED_LABELS) | frozenset({
    "PASSPORT", "DRIVER_LICENSE", "REGION", "BUILDING", "PUBLIC_PERSON", "PUBLIC_ADDRESS"
})



def _digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choices(string.digits, k=n))


def _person(rng: random.Random, split: str) -> str:
    last = rng.choice(SURNAME[split])
    if rng.random() < 0.45:
        first, patr = rng.choice(FIRST_F), rng.choice(PATR_F)
        if not last.endswith("а"):
            last += "а"
    else:
        first, patr = rng.choice(FIRST_M), rng.choice(PATR_M)
    if rng.random() < 0.16:
        return f"{last} {first[0]}. {patr[0]}."
    return f"{last} {first} {patr}"


def _date(rng: random.Random, *, issue: bool = False) -> str:
    d, m = rng.randint(1, 28), rng.randint(1, 12)
    y = rng.randint(2012, 2025) if issue else rng.randint(1955, 2004)
    style = rng.randrange(6)
    return [
        f"{d:02}.{m:02}.{y}", f"{y}-{m:02}-{d:02}", f"{m:02}/{d:02}/{y}",
        f"{y}.{d:02}.{m:02}", f"{d} {MONTHS[m-1]} {y} года", f"{d:02} {m:02} {y}",
    ][style]


def _inn12(rng: random.Random) -> str:
    a = [int(x) for x in _digits(rng, 10)]
    a.append(sum(x*y for x, y in zip(a, [7,2,4,10,3,5,9,4,6,8])) % 11 % 10)
    a.append(sum(x*y for x, y in zip(a, [3,7,2,4,10,3,5,9,4,6,8])) % 11 % 10)
    return "".join(map(str, a))


def _pan(rng: random.Random) -> str:
    # Test PAN-like number generated locally; no claim that it is unassigned.
    a = [int(x) for x in "4" + _digits(rng, 14)]
    total = sum((v*2-9 if v*2 > 9 else v*2) if i % 2 == 0 else v for i, v in enumerate(a))
    a.append((-total) % 10)
    raw = "".join(map(str, a))
    sep = rng.choice(["", " ", "-", "\u00a0"])
    return sep.join(raw[i:i+4] for i in range(0, 16, 4))


class Builder:
    def __init__(self) -> None:
        self.text = ""
        self.entities: list[dict[str, Any]] = []

    def add(self, value: str) -> None:
        self.text += value

    def entity(self, value: str, label: str, *, privacy: str = "personal", subject: str | None = "s1") -> tuple[int, int]:
        start = len(self.text)
        self.text += value
        end = len(self.text)
        self.entities.append({"start": start, "end": end, "label": label, "text": value,
                              "privacy": privacy, "subject_id": subject})
        return start, end

    def add_nested(self, start: int, end: int, label: str, *, privacy: str = "personal", subject: str | None = "s1") -> None:
        self.entities.append({"start": start, "end": end, "label": label, "text": self.text[start:end],
                              "privacy": privacy, "subject_id": subject})

    def field(self, prefix: str, value: str, label: str, suffix: str = "; ", **kwargs: Any) -> None:
        self.add(prefix)
        self.entity(value, label, **kwargs)
        self.add(suffix)

    def person(self, value: str, **kwargs: Any) -> None:
        self.entity(value, "PERSON", **kwargs)

    def passport(self, rng: random.Random, *, subject: str = "s1") -> None:
        self.add(rng.choice(["Паспорт РФ: ", "паспорт, ", "Документ: паспорт РФ, "]))
        coarse_start = len(self.text)
        self.add("серия ")
        series = _digits(rng, 4)
        if rng.random() < .45:
            series = series[:2] + " " + series[2:]
        self.entity(series, "PASSPORT_SERIES", subject=subject)
        self.add(rng.choice([" номер ", " № ", ", номер: "]))
        number = _digits(rng, 6)
        if rng.random() < .35:
            number = number[:3] + " " + number[3:]
        self.entity(number, "PASSPORT_NUMBER", subject=subject)
        coarse_end = len(self.text)
        self.add_nested(coarse_start, coarse_end, "PASSPORT", subject=subject)
        self.add("; ")
        self.field("Кем выдан: ", rng.choice(ISSUERS), "PASSPORT_ISSUER", subject=subject)
        self.field("Код подразделения: ", _digits(rng, 3) + rng.choice(["-", " ", "‑"]) + _digits(rng, 3), "PASSPORT_DIVISION_CODE", subject=subject)
        self.field("Дата выдачи: ", _date(rng, issue=True), "PASSPORT_ISSUE_DATE", subject=subject)

    def driver_license(self, rng: random.Random, *, subject: str = "s1") -> None:
        self.add(rng.choice(["Водительское удостоверение: ", "В/У: ", "Права: "]))
        coarse_start = len(self.text)
        self.add("серия ")
        self.entity(_digits(rng, 4), "DRIVER_LICENSE_SERIES", subject=subject)
        self.add(rng.choice([" номер ", " № "]))
        self.entity(_digits(rng, 6), "DRIVER_LICENSE_NUMBER", subject=subject)
        coarse_end = len(self.text)
        self.add_nested(coarse_start, coarse_end, "DRIVER_LICENSE", subject=subject)
        self.add("; ")

    def short_address(self, rng: random.Random, *, privacy: str = "personal", subject: str | None = "s1") -> None:
        city, street, house = rng.choice(CITIES), rng.choice(STREETS), str(rng.randint(1, 180))
        start = len(self.text)
        self.entity(city, "CITY", privacy=privacy, subject=subject)
        self.add(", ")
        self.entity(street, "STREET", privacy=privacy, subject=subject)
        self.add(" ")
        self.entity(house, "HOUSE", privacy=privacy, subject=subject)
        end = len(self.text)
        self.add_nested(start, end, "ADDRESS", privacy=privacy, subject=subject)
        if privacy == "public":
            self.add_nested(start, end, "PUBLIC_ADDRESS", privacy="public", subject=None)

    def full_address_fields(self, rng: random.Random, *, subject: str = "s1") -> None:
        self.field("Страна: ", rng.choice(["Россия", "РФ"]), "COUNTRY", subject=subject)
        self.field("Индекс: ", str(rng.randint(100000, 699999)), "POSTAL_CODE", subject=subject)
        self.field("Регион: ", rng.choice(REGIONS), "REGION", subject=subject)
        self.add("Адрес: ")
        self.short_address(rng, subject=subject)
        self.add("; ")
        self.field("Корпус: ", str(rng.randint(1, 8)), "BUILDING", subject=subject)
        self.field("Квартира: ", str(rng.randint(1, 450)), "APARTMENT", subject=subject)

    def card(self, rng: random.Random, *, subject: str = "s1") -> None:
        self.field("Номер карты: ", _pan(rng), "CARD_NUMBER", subject=subject)
        self.field("Имя держателя: ", rng.choice(["IVAN PETROV", "ANNA ORLOVA", "SERGEY VOLKOV", "MARIA SOKOLOVA"]), "CARDHOLDER", subject=subject)
        self.field(rng.choice(["CVV: ", "CVC: "]), _digits(rng, 3), "CVV", subject=subject)
        self.field(rng.choice(["ПИН: ", "PIN-код: ", "Пин-код карты: "]), _digits(rng, 4), "PIN", subject=subject)

    def casefold_variant(self, mode: str) -> None:
        fn = {"original": lambda x: x, "lower": str.lower, "upper": str.upper}[mode]
        starts = [0]
        chunks, n = [], 0
        for ch in self.text:
            part = fn(ch)
            chunks.append(part)
            n += len(part)
            starts.append(n)
        transformed = "".join(chunks)
        for e in self.entities:
            e["start"], e["end"] = starts[e["start"]], starts[e["end"]]
            e["text"] = transformed[e["start"]:e["end"]]
        self.text = transformed


def build_record(split: str, index: int, rng: random.Random) -> dict[str, Any]:
    b = Builder()
    subject = f"supp:{split}:{index:06d}"
    name = _person(rng, split)
    scenario = index % 11

    if scenario == 0:
        b.add("Анкета клиента. ФИО: "); b.person(name)
        b.add("; "); b.field("Дата рождения: ", _date(rng), "BIRTH_DATE")
        b.field("Место рождения: ", rng.choice([f"г. {rng.choice(CITIES)}", rng.choice(CITIES)]), "BIRTH_PLACE")
        b.field("Гражданство: ", rng.choice(CITIZENSHIPS), "CITIZENSHIP")
    elif scenario == 1:
        b.add("Для идентификации клиента "); b.person(name); b.add(" нужны реквизиты. ")
        b.passport(rng)
    elif scenario == 2:
        b.add("Заявитель "); b.person(name); b.add(" сообщил данные документа. ")
        b.driver_license(rng)
    elif scenario == 3:
        b.add("Адрес регистрации клиента: "); b.short_address(rng); b.add("; ")
        b.full_address_fields(rng)
    elif scenario == 4:
        b.add("Контакты клиента "); b.person(name); b.add(": ")
        digits = _digits(rng, 10)
        phone = rng.choice([f"+7 ({digits[:3]}) {digits[3:6]}-{digits[6:8]}-{digits[8:]}", "+7" + digits, "8 " + digits[:3] + " " + digits[3:6] + " " + digits[6:8] + " " + digits[8:]])
        b.field("Телефон: ", phone, "PHONE")
        b.field("Email: ", f"client{index}.{split}@example.org", "EMAIL")
        b.field("ИНН физлица: ", _inn12(rng), "INN")
    elif scenario == 5:
        b.add("Платёжные реквизиты владельца "); b.person(name); b.add(": ")
        b.card(rng)
    elif scenario == 6:
        b.add("Карточка клиента: "); b.person(name); b.add("; ")
        b.field("Рождён: ", _date(rng), "BIRTH_DATE")
        b.field("Место рождения: ", rng.choice(CITIES), "BIRTH_PLACE")
        b.field("Гражданство: ", rng.choice(CITIZENSHIPS), "CITIZENSHIP")
        b.passport(rng)
        b.driver_license(rng)
    elif scenario == 7:
        b.add("Профиль доставки. Получатель: "); b.person(name); b.add("; ")
        b.full_address_fields(rng)
        b.add("Телефон: "); b.entity("+7" + _digits(rng, 10), "PHONE"); b.add("; ")
        b.add("Почта: "); b.entity(f"delivery{index}@example.net", "EMAIL"); b.add(". ")
    elif scenario == 8:
        # Public references are context supervision, not a whitelist by string.
        b.add("Для справки расскажите о поэте ")
        start, end = b.entity(rng.choice(["Александре Пушкине", "Михаиле Лермонтове", "Антоне Чехове"]), "PERSON", privacy="public", subject=None)
        b.add_nested(start, end, "PUBLIC_PERSON", privacy="public", subject=None)
        b.add(". Адрес отделения банка: "); b.short_address(rng, privacy="public", subject=None)
        b.add(". Личные данные клиента здесь не указаны.")
    elif scenario == 9:
        # Hard negatives: all task labels are exhaustively annotated, with no positive spans.
        b.add(f"Заказ № {_digits(rng, 6)}, код товара {_digits(rng, 4)}. ")
        b.add(f"Совещание назначено на {_date(rng, issue=True)}. ")
        b.add("Фраза «пин-код карты» приведена как термин без значения. ")
        b.add(f"Сумма операции {rng.randint(100, 90000)} рублей.")
    else:
        b.add("В обращении два субъекта. Первый: "); b.person(name, subject="s1"); b.add("; ")
        b.passport(rng, subject="s1")
        second = _person(rng, split)
        b.add("Второй: "); b.person(second, subject="s2"); b.add("; ")
        b.driver_license(rng, subject="s2")
        b.add("Его карта: "); b.entity(_pan(rng), "CARD_NUMBER", subject="s2"); b.add("; ")
        b.add("CVV: "); b.entity(_digits(rng, 3), "CVV", subject="s2"); b.add(". ")

    b.add(f"\nУчебный идентификатор {split}-{index:06d}.")
    mode = rng.choices(["original", "lower", "upper"], weights=[7, 2, 1])[0]
    b.casefold_variant(mode)
    row = {
        "id": f"ru-task-supplement-{split}-{index:06d}",
        "text": b.text,
        "entities": b.entities,
        "annotated_labels": sorted(SUPPLEMENT_LABELS),
        "language": "ru",
        "source": "task-synthetic-supplement-v2",
        "split": split,
        "synthetic": True,
        "provenance": "locally generated non-issued values; exact offsets by construction",
        "annotation_method": "construction_offsets",
        "license": "CC0-1.0",
        "scenario": scenario,
        "synthetic_subjects": [subject],
    }
    validate_record(row)
    return row


def make_splits(*, seed: int = 20260922, sizes: dict[str, int] | None = None) -> dict[str, list[dict[str, Any]]]:
    sizes = sizes or {"train": 12000, "dev": 1500, "test": 1500}
    out: dict[str, list[dict[str, Any]]] = {}
    for split_index, (split, size) in enumerate(sizes.items()):
        out[split] = [build_record(split, i, random.Random(seed + split_index * 1_000_000 + i)) for i in range(size)]
    return out


def audit(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {"schema": "task-supplement-v2", "splits": {}}
    for split, rows in splits.items():
        counts = Counter(e["label"] for row in rows for e in row["entities"])
        required = {label: counts.get(label, 0) for label in sorted(TASK_REQUIRED_LABELS)}
        result["splits"][split] = {
            "records": len(rows),
            "entities": sum(counts.values()),
            "by_label": dict(sorted(counts.items())),
            "required_task_coverage": required,
            "missing_required": [k for k, v in required.items() if v == 0],
        }
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/supplement")
    p.add_argument("--train", type=int, default=12000)
    p.add_argument("--dev", type=int, default=1500)
    p.add_argument("--test", type=int, default=1500)
    p.add_argument("--seed", type=int, default=20260922)
    args = p.parse_args()
    if min(args.train, args.dev, args.test) < 110:
        p.error("each split must contain at least 110 rows so all scenarios repeat")
    splits = make_splits(seed=args.seed, sizes={"train": args.train, "dev": args.dev, "test": args.test})
    report = audit(splits)
    for split, rows in splits.items():
        write_jsonl(Path(args.output) / f"{split}.jsonl", rows)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
