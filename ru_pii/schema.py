"""Canonical annotation schema. Character intervals are [start, end)."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any

# Short English prompts intentionally stable across training and inference.
LABELS: dict[str, str] = {
    "PERSON": "person",
    "BIRTH_DATE": "date of birth",
    "BIRTH_PLACE": "place of birth",
    # Coarse document spans are useful for public corpora that annotate the whole identifier.
    "PASSPORT": "passport series and number",
    "PASSPORT_SERIES": "passport series",
    "PASSPORT_NUMBER": "passport number",
    "CITIZENSHIP": "citizenship",
    "PASSPORT_ISSUER": "passport issuing authority",
    "PASSPORT_DIVISION_CODE": "passport department code",
    "PASSPORT_ISSUE_DATE": "passport issue date",
    "DRIVER_LICENSE": "driving license series and number",
    "DRIVER_LICENSE_SERIES": "driving license series",
    "DRIVER_LICENSE_NUMBER": "driving license number",
    "ADDRESS": "address",
    "COUNTRY": "country",
    "POSTAL_CODE": "postal code",
    "REGION": "region",
    "DISTRICT": "district",
    "CITY": "city",
    "STREET": "street",
    "HOUSE": "house number",
    "BUILDING": "building number",
    "APARTMENT": "apartment number",
    "EMAIL": "email",
    "PHONE": "phone number",
    "INN": "tax identification number",
    "CARD_NUMBER": "bank card number",
    "CVV": "card security code",
    "PIN": "card pin code",
    "CARDHOLDER": "cardholder name",
    "OTHER_ID": "other identity document number",
    # Useful bonus PII/document classes available in the real/mixed Russian corpus.
    "SNILS": "SNILS insurance account number",
    "OMS": "OMS medical insurance policy number",
    "MILITARY_ID": "military identity document number",
    "BIRTH_CERTIFICATE": "birth certificate number",
    "IP_ADDRESS": "IP address",
    "URL": "URL",
    # Auxiliary context signals, never an automatic public-data whitelist.
    "PUBLIC_PERSON": "public person reference",
    "PUBLIC_ADDRESS": "public institution address",
}
AUXILIARY = frozenset({"PUBLIC_PERSON", "PUBLIC_ADDRESS"})

# Labels required by the hackathon task.  PASSPORT/DRIVER_LICENSE are coarse
# convenience labels; the task-specific fine-grained series/number classes are
# also trained and validated.  REGION/DISTRICT/BUILDING are additional address
# detail labels, not substitutes for the explicitly required parts.
TASK_REQUIRED_LABELS = frozenset({
    "PERSON", "BIRTH_DATE", "BIRTH_PLACE",
    "PASSPORT_SERIES", "PASSPORT_NUMBER", "CITIZENSHIP",
    "PASSPORT_ISSUER", "PASSPORT_DIVISION_CODE", "PASSPORT_ISSUE_DATE",
    "DRIVER_LICENSE_SERIES", "DRIVER_LICENSE_NUMBER",
    "ADDRESS", "COUNTRY", "POSTAL_CODE", "CITY", "STREET", "HOUSE", "APARTMENT",
    "EMAIL", "PHONE", "INN", "CARD_NUMBER", "CVV", "PIN", "CARDHOLDER",
})
COARSE_DOCUMENT_LABELS = frozenset({"PASSPORT", "DRIVER_LICENSE"})
EXTRA_LABELS = frozenset({"REGION", "DISTRICT", "BUILDING", "OTHER_ID", "SNILS", "OMS", "MILITARY_ID", "BIRTH_CERTIFICATE", "IP_ADDRESS", "URL"})
ADDRESS_PARTS = frozenset({"COUNTRY", "POSTAL_CODE", "REGION", "DISTRICT", "CITY", "STREET", "HOUSE", "BUILDING", "APARTMENT"})
PRIVACY_VALUES = frozenset({"personal", "public", "unknown"})

@dataclass(frozen=True)
class Entity:
    start: int
    end: int
    label: str
    score: float
    text: str
    source: str = "gliner"
    privacy_hint: str = "personal_or_unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
