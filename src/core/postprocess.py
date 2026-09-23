"""Post-processing: filter out public figures and public addresses.

The task requires distinguishing "поэт Александр Пушкин" (public figure, NOT
personal data) from a bank client. NER models (GLiNER and distilled students)
tend to tag any PERSON as personal data. This module applies context heuristics
to demote such spans to non-PII.

Heuristics:
  * A PERSON preceded by a public-figure cue ("поэт", "писатель", "президент",
    "актёр", "певец", "учёный", "министр", "депутат", "художник", "композитор",
    "режиссёр", "спортсмен", "космонавт", "царь", "король", "император") is a
    public figure, not personal data.
  * An ADDRESS/CITY/STREET preceded by "отделение банка", "банк", "офис",
    "филиал", "адрес отделения" is a public institution address, not personal.
"""
from __future__ import annotations

import re
from typing import Sequence

from ru_pii.schema import Entity

# Cues that mark a PERSON as a public figure (not personal data).
_PUBLIC_FIGURE_CUES = re.compile(
    r"(поэт|писатель|президент|акт[её]р|певец|уч[её]ный|министр|депутат|художник|"
    r"композитор|режисс[её]р|спортсмен|космонавт|царь|король|император|генерал|"
    r"маршал|академик|профессор|доктор|премьер|канцлер|королева|принц|князь|"
    r"граф|барон|сенатор|губернатор|мэр|глава|лидер|основатель|изобретатель|"
    r"композитор|драматург|философ|историк|журналист|ведущий|блогер|"
    r"футболист|хоккеист|теннисист|бокс[её]р|шахматист|пианист|скрипач|"
    r"дириж[её]р|балетмейстер|архитектор|скульптор|фотограф|модельер|"
    r"дизайнер|программист|инженер|врач|хирург|адвокат|судья|прокурор|"
    r"посол|консул|дипломат|астронавт|л[её]тчик|моряк|генерал-полковник)",
    re.IGNORECASE,
)

# Cues that mark an ADDRESS as a public institution (not personal data).
_PUBLIC_ADDRESS_CUES = re.compile(
    r"(отделение банка|банк|офис|филиал|адрес отделения|головной офис|"
    r"представительство|отделение|клиника|больница|школа|университет|"
    r"институт|академия|театр|музей|библиотека|стадион|аэропорт|вокзал|"
    r"администрация|правительство|министерство|дума|совет|посольство|"
    r"консульство|завод|фабрика|компания|корпорация|фирма|магазин|"
    r"ресторан|кафе|гостиница|отель)",
    re.IGNORECASE,
)

# Address labels that may be public institution addresses.
_ADDRESS_LABELS = frozenset({"ADDRESS", "CITY", "STREET", "HOUSE", "BUILDING", "COUNTRY", "REGION"})


def _is_public_figure(text: str, entity: Entity) -> bool:
    """Check if a PERSON is preceded by a public-figure cue."""
    before = text[max(0, entity.start - 60):entity.start]
    return bool(_PUBLIC_FIGURE_CUES.search(before))


def _is_public_address(text: str, entity: Entity) -> bool:
    """Check if an address is preceded by a public-institution cue."""
    before = text[max(0, entity.start - 60):entity.start]
    return bool(_PUBLIC_ADDRESS_CUES.search(before))


def filter_public(text: str, entities: Sequence[Entity]) -> list[Entity]:
    """Remove entities that are public figures / public addresses (not PII)."""
    out: list[Entity] = []
    for e in entities:
        if e.label == "PERSON" and _is_public_figure(text, e):
            continue
        if e.label in _ADDRESS_LABELS and _is_public_address(text, e):
            continue
        out.append(e)
    return out