#!/usr/bin/env python3
"""Generate the public-person hard-negative supplement.

Produces data/public_negatives/{train,dev,test}.jsonl with three kinds of rows:
pure negatives (a famous public figure in a neutral context, nothing to mask),
mixed rows (a public figure alongside real PII that must still be masked), and
bank-branch address negatives (PUBLIC_ADDRESS with nested CITY/STREET/HOUSE).
Templates use cased forms of famous surnames and random case variants to make
detection register-invariant. Deterministic under SEED.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_persons import (
    CITIES,
    FAMOUS_LAST_NAMES,
    NON_DECLINABLE,
    ORDINARY_FIRST_F,
    ORDINARY_FIRST_M,
    ORDINARY_SURNAMES,
    PAINTER_ROLES,
    PATRONYMICS_F,
    PATRONYMICS_M,
    POET_ROLES,
    PUBLIC_PERSONS,
    ROLES,
    SCI_ROLES,
    STREETS,
    WRITER_ROLES,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "public_negatives"
SOURCE_NAME = "task-synthetic-public-negatives-v1"
SEED = 20260922
TOTAL = 2400
KIND_WEIGHTS = [("pure", 55), ("mixed", 33), ("bank", 12)]
CASE_WEIGHTS = [("keep", 60), ("lower", 25), ("upper", 15)]

COMPOSER_ROLES = {"композитор"}
WORD_ROLES = POET_ROLES | WRITER_ROLES | SCI_ROLES | PAINTER_ROLES | COMPOSER_ROLES


def decline_one(s: str, case: str, gender: str) -> str:
    if case == "nom" or s in NON_DECLINABLE:
        return s
    if gender == "f":
        if s.endswith("ая"):
            return s[:-2] + "ой"
        if s.endswith(("ова", "ева", "ёва", "ина", "ына")):
            return s[:-1] + "ой"
        return s
    if s.endswith(("ский", "цкий")):
        b = s[:-2]
        return b + {"gen": "ого", "dat": "ому", "acc": "ого", "ins": "им", "prep": "ом"}[case]
    if s.endswith(("ой", "ый", "ий")):
        b = s[:-2]
        return b + {"gen": "ого", "dat": "ому", "acc": "ого", "ins": "ым", "prep": "ом"}[case]
    if s.endswith(("ов", "ев", "ёв", "ин", "ын")):
        return s + {"gen": "а", "dat": "у", "acc": "а", "ins": "ым", "prep": "е"}[case]
    if s.endswith("а"):
        b = s[:-1]
        return b + {"gen": "и", "dat": "е", "acc": "у", "ins": "ой", "prep": "е"}[case]
    if s.endswith("ь"):
        b = s[:-1]
        return b + {"gen": "я", "dat": "ю", "acc": "я", "ins": "ем", "prep": "е"}[case]
    if s[-1:].lower() in "жшчщц":
        return s + {"gen": "а", "dat": "у", "acc": "а", "ins": "ем", "prep": "е"}[case]
    return s + {"gen": "а", "dat": "у", "acc": "а", "ins": "ом", "prep": "е"}[case]


def decline(surname: str, case: str, gender: str) -> str:
    return "-".join(decline_one(part, case, gender) for part in surname.split("-"))


PURE = [
    ("Расскажи, пожалуйста, {role_prep} {fam_prep}.", None, None),
    ("Что ты знаешь {role_prep} {fam_prep}?", None, None),
    ("Стихи {fam_gen} изучают в школе.", None, POET_ROLES),
    ("Романы {fam_gen} переведены на десятки языков.", None, WRITER_ROLES),
    ("На уроке литературы разбирали творчество {fam_gen}.", None, WRITER_ROLES),
    ("Пользователь просит подготовить доклад {role_prep} {fam_prep}.", None, None),
    ("Памятник {fam_dat} стоит в центре города.", None, None),
    ("В музее открылась выставка, посвящённая {fam_dat}.", None, None),
    ("Объясни ребёнку, кто такой {full_nom}.", "m", None),
    ("Объясни ребёнку, кто такая {full_nom}.", "f", None),
    ("Цитаты {fam_gen} актуальны до сих пор.", None, WORD_ROLES),
    ("Сравни творчество {fam_gen} и {fam2_gen}.", None, WORD_ROLES),
    ("Кратко расскажи биографию {fam_gen}.", None, None),
    ("Какое влияние оказал {fam_nom} на культуру?", "m", None),
    ("Какое влияние оказала {fam_nom} на культуру?", "f", None),
    ("Портрет {fam_gen} висит в главной галерее.", None, PAINTER_ROLES),
    ("Посоветуй произведения {fam_gen} для внеклассного чтения.", None, WRITER_ROLES),
    ("Интересные факты из жизни {fam_gen}.", None, None),
    ("Документальный фильм {role_prep} {fam_prep} вышел на прошлой неделе.", None, None),
    ("Найди информацию о детстве и юности {fam_gen}.", None, None),
    ("Взгляды {fam_gen} на искусство повлияли на целую эпоху.", None, None),
    ("Закажи билеты на спектакль по произведению {fam_gen}.", None, WRITER_ROLES),
    ("Книги {fam_gen} продаются в каждом магазине.", None, WRITER_ROLES),
    ("Школьники готовят проект {role_prep} {fam_prep}.", None, None),
    ("Подбери факты для стенгазеты {role_prep} {fam_prep}.", None, None),
    ("Спектакль по пьесе {fam_gen} идёт в театре.", None, WRITER_ROLES),
    ("Составь план экскурсии по местам, связанным с {fam_ins}.", None, None),
    ("Расскажи смешную историю про {fam_acc}.", None, None),
    ("Чем известен {fam_nom}?", "m", None),
    ("Чем известна {fam_nom}?", "f", None),
    ("Какие открытия сделал {fam_nom}?", None, SCI_ROLES),
    ("Напиши сочинение по картине {fam_gen}.", None, PAINTER_ROLES),
    ("Пользователь спрашивает: чем знаменит {full_nom}?", "m", None),
    ("Пользователь спрашивает: чем знаменита {full_nom}?", "f", None),
    ("Музыка {fam_gen} звучит в фильме.", None, COMPOSER_ROLES),
]

MIXED = [
    ("Расскажи {role_prep} {fam_prep}. Заодно проверь мои реквизиты: карта {card}, ИНН {inn}.", None, None),
    ("Мне нравится {fam_nom}. Отправь мою анкету: {person}, дата рождения {birthdate}, {phone}.", None, None),
    ("Доклад о {fam_prep} нужно прислать с адреса {email} не позднее завтрашнего дня.", None, None),
    ("Сравни {fam_acc} с современниками. Мои контакты: {phone}, {email}.", None, None),
    ("Готовлю реферат о {fam_dat} по литературе. Мой ИНН для пропуска: {inn}.", None, None),
    ("Пользователь {person} спрашивает {role_prep} {fam_prep}.", None, None),
    ("Оформляю подписку на журнал {role_prep} {fam_prep} от имени {person}, почта {email}.", None, None),
    ("Закажи книгу {fam_gen} на адрес: {city}, {street} {house}.", None, WRITER_ROLES),
    ("Подскажи, где почитать {role_prep} {fam_prep}, и сохрани мой телефон {phone}.", None, None),
    ("Нравится ли тебе {fam_nom}? Вот email для рассылки: {email}.", None, None),
    ("Перечисли заодно мои данные: {person}, {birthdate}, карта {card}.", None, None),
    ("Расскажи {role_prep} {fam_prep} и подскажи отделение банка: {bank_addr}. Мой телефон {phone}.", None, None),
    ("Помоги оформить цитатник {fam_gen}. Мой контакт: {phone}, {email}.", None, WORD_ROLES),
]

BANK = [
    "Ближайшее отделение банка находится по адресу: {bank_addr}.",
    "Подскажи отделение банка: {bank_addr}.",
    "Головной офис банка: {bank_addr}, работает до 20:00.",
    "Куда отправить письмо? В банк по адресу {bank_addr}.",
    "Запиши адрес филиала: {bank_addr}.",
    "Банкомат этого банка: {bank_addr}.",
    "Офис обслуживания клиентов: {bank_addr}.",
    "Найди банкомат по адресу {bank_addr}.",
    "Отделение для юридических лиц: {bank_addr}.",
    "Центральный офис банка расположен по адресу {bank_addr}.",
]

TOKEN_RE = re.compile(r"\{(\w+)\}")
PII_LABELS = {
    "person": "PERSON",
    "card": "CARD_NUMBER",
    "inn": "INN",
    "phone": "PHONE",
    "email": "EMAIL",
    "birthdate": "BIRTH_DATE",
    "city": "CITY",
    "street": "STREET",
    "house": "HOUSE",
}


def gen_pii(rng):
    male = rng.random() < 0.5
    first = rng.choice(ORDINARY_FIRST_M if male else ORDINARY_FIRST_F)
    patr = rng.choice(PATRONYMICS_M if male else PATRONYMICS_F)
    last = rng.choice(ORDINARY_SURNAMES)
    assert last.lower() not in FAMOUS_LAST_NAMES
    if not male:
        last = last + "а"
    person = f"{first} {patr} {last}" if rng.random() < 0.7 else f"{first} {last}"
    return {
        "person": person,
        "card": " ".join(str(rng.randint(1000, 9999)) for _ in range(4)),
        "inn": str(rng.randrange(10**11, 10**12)),
        "phone": f"+7 9{rng.randint(10, 99)} {rng.randint(100, 999)}-{rng.randint(10, 99)}-{rng.randint(10, 99)}",
        "email": f"{rng.choice(['ivanov', 'petrova', 'sidorova', 'smirnova'])}{rng.randint(1960, 2005)}@example.com",
        "birthdate": f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.{rng.randint(1950, 2005)}",
        "city": rng.choice(CITIES),
        "street": rng.choice(STREETS),
        "house": str(rng.randint(1, 120)),
        "bank_city": rng.choice(CITIES),
        "bank_street": rng.choice(STREETS),
        "bank_house": str(rng.randint(1, 120)),
    }


def make_resolver(person, person2, pii):
    def resolve(placeholder):
        if placeholder == "full_nom":
            value = f"{person[0]} {person[1]}"
            return value, [(0, len(value), "PUBLIC_PERSON")]
        if placeholder == "role_prep":
            return ROLES[person[2]]["prep"], None
        if placeholder == "role_nom":
            return ROLES[person[2]]["nom"], None
        if placeholder == "fam2_gen":
            value = decline(person2[1], "gen", person2[3])
            return value, [(0, len(value), "PUBLIC_PERSON")]
        if placeholder.startswith("fam_"):
            case = placeholder.split("_", 1)[1]
            value = decline(person[1], case, person[3])
            return value, [(0, len(value), "PUBLIC_PERSON")]
        if placeholder == "bank_addr":
            city = pii["bank_city"]
            street = pii["bank_street"]
            house = pii["bank_house"]
            value = f"{city}, {street} {house}"
            labels = [
                (0, len(value), "PUBLIC_ADDRESS"),
                (0, len(city), "CITY"),
                (len(city) + 2, len(city) + 2 + len(street), "STREET"),
                (len(city) + 2 + len(street) + 1, len(value), "HOUSE"),
            ]
            return value, labels
        if placeholder in PII_LABELS:
            value = pii[placeholder]
            return value, [(0, len(value), PII_LABELS[placeholder])]
        raise KeyError(placeholder)

    return resolve


def fill(template: str, resolver):
    out = []
    spans = []
    cursor = 0
    pos = 0
    for match in TOKEN_RE.finditer(template):
        literal = template[pos:match.start()]
        out.append(literal)
        cursor += len(literal)
        value, labels = resolver(match.group(1))
        if labels:
            for start, end, label in labels:
                spans.append((cursor + start, cursor + end, label))
        out.append(value)
        cursor += len(value)
        pos = match.end()
    out.append(template[pos:])
    return "".join(out), spans


def apply_case(text: str, mode: str) -> str:
    if mode == "lower":
        return text.lower()
    if mode == "upper":
        return text.upper()
    return text


def validate(text: str, spans: list[tuple[int, int, str]]) -> None:
    spans = sorted(spans, key=lambda x: (x[0], x[0] - x[1]))
    open_ends: list[int] = []
    for start, end, _ in spans:
        while open_ends and open_ends[-1] <= start:
            open_ends.pop()
        if open_ends and end > open_ends[-1]:
            raise AssertionError(f"crossing spans: {spans}")
        open_ends.append(end)
    for start, end, label in spans:
        entity_text = text[start:end]
        assert entity_text, f"empty span {start}-{end} {label}"


def split_of(text: str) -> str:
    digest = hashlib.md5((str(SEED) + text).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 2**32
    if bucket < 0.80:
        return "train"
    if bucket < 0.90:
        return "dev"
    return "test"


def main() -> int:
    rng = random.Random(SEED)
    rows_by_split = {"train": [], "dev": [], "test": []}
    seen: set[str] = set()
    kinds = Counter()
    label_totals: Counter = Counter()
    attempts = 0

    while sum(len(v) for v in rows_by_split.values()) < TOTAL:
        attempts += 1
        if attempts > TOTAL * 20:
            print("ERROR: could not generate unique rows", file=sys.stderr)
            return 1
        kind = rng.choices([k for k, _ in KIND_WEIGHTS], weights=[w for _, w in KIND_WEIGHTS])[0]
        person = rng.choice(PUBLIC_PERSONS)
        person2 = rng.choice(PUBLIC_PERSONS)
        pii = gen_pii(rng)
        if kind == "pure":
            candidates = [t for t in PURE if (t[1] is None or t[1] == person[3]) and (t[2] is None or person[2] in t[2])]
            template = rng.choice(candidates)[0]
        elif kind == "mixed":
            candidates = [t for t in MIXED if (t[1] is None or t[1] == person[3]) and (t[2] is None or person[2] in t[2])]
            template = rng.choice(candidates)[0]
        else:
            template = rng.choice(BANK)
        text, spans = fill(template, make_resolver(person, person2, pii))
        case_mode = rng.choices([k for k, _ in CASE_WEIGHTS], weights=[w for _, w in CASE_WEIGHTS])[0]
        text = apply_case(text, case_mode)
        if text in seen:
            continue
        seen.add(text)
        validate(text, spans)
        split = split_of(text)
        entities = [
            {"start": start, "end": end, "label": label, "text": text[start:end]}
            for start, end, label in sorted(spans, key=lambda x: (x[0], x[0] - x[1]))
        ]
        kinds[kind] += 1
        label_totals.update(e["label"] for e in entities)
        rows_by_split[split].append((text, entities))

    for split, rows in rows_by_split.items():
        path = OUT_DIR / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for n, (text, entities) in enumerate(rows, start=1):
                record = {
                    "id": f"task-pubneg-{split}-{n:06d}",
                    "text": text,
                    "split": split,
                    "source": SOURCE_NAME,
                    "annotated_labels": sorted({e["label"] for e in entities}),
                    "entities": entities,
                    "language": "ru",
                    "synthetic": True,
                    "provenance": "generated hard negatives: public figures and bank branch addresses",
                    "source_url": "local generator scripts/make_public_negatives.py",
                    "annotation_method": "template generation with tracked char offsets",
                    "license": "generated for this project",
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": "ru-pii-public-negatives-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "total": TOTAL,
        "kind_weights": dict(KIND_WEIGHTS),
        "audit": {
            "splits": {
                split: {
                    "records": len(rows),
                    "entities": sum(len(e) for _, e in rows),
                    "by_label": dict(sorted(Counter(
                        ent["label"] for _, ents in rows for ent in ents
                    ).items())),
                    "by_source": {SOURCE_NAME: len(rows)},
                }
                for split, rows in rows_by_split.items()
            },
            "kinds": dict(kinds),
            "checks_passed": True,
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = f"""# Public hard negatives (generated)

Generated by scripts/make_public_negatives.py from scripts/public_persons.py:
{len(PUBLIC_PERSONS)} famous public figures (writers, scientists, composers,
painters, cosmonauts, pre-revolutionary statesmen) in {len(PURE) + len(MIXED) + len(BANK)} varied templates.

Row kinds:
- pure ({kinds['pure']}): famous person in a neutral context, nothing to mask (PUBLIC_PERSON).
- mixed ({kinds['mixed']}): famous person + real PII (PERSON, CARD_NUMBER, INN, PHONE, EMAIL,
  BIRTH_DATE, CITY/STREET/HOUSE) that must still be masked.
- bank ({kinds['bank']}): bank branch address (PUBLIC_ADDRESS with nested CITY/STREET/HOUSE).

Case variants: keep/lower/upper 60/25/15. Deterministic split 80/10/10 by md5.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    print(f"kinds: {dict(kinds)}")
    for split, rows in rows_by_split.items():
        print(f"{split}: {len(rows)} records, {sum(len(e) for _, e in rows)} entities")
    print("labels:", dict(label_totals))
    print("\nexamples:")
    shown = 0
    for split, rows in rows_by_split.items():
        for text, entities in rows:
            if entities and any(e["label"] == "PERSON" for e in entities) and any(e["label"] == "PUBLIC_PERSON" for e in entities) and shown < 4:
                shown += 1
                print(f"  TEXT: {text[:160]}")
                print(f"  SPANS: {[(text[e['start']:e['end']], e['label']) for e in entities]}")
    print(f"\nwritten to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
