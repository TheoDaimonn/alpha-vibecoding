"""Filter explicit public annotations without guessing privacy from occupations.

Words such as 'doctor', 'bank' or 'office' do not prove that nearby personal
names or home addresses are public. Only model annotations are used here.
"""
from __future__ import annotations

from collections.abc import Sequence

from ru_pii.schema import AUXILIARY, Entity


def filter_public(text: str, entities: Sequence[Entity]) -> list[Entity]:
    """Drop auxiliary public labels; retain independent personal predictions."""
    return [entity for entity in entities if entity.label not in AUXILIARY]
