"""Canonical annotation schema. Character intervals are [start, end)."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json

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


def validate_record(record: dict[str, Any]) -> None:
    """Reject corrupted offsets; do not 'repair' them with a global string search."""
    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("record.text must be a non-empty string")
    if not isinstance(record.get("id"), str) or not record["id"]:
        raise ValueError("record.id is required")
    labels = record.get("annotated_labels")
    if not isinstance(labels, list) or not labels or not set(labels) <= LABELS.keys():
        raise ValueError("annotated_labels must list exhaustively annotated canonical types")
    entities = record.get("entities")
    if not isinstance(entities, list):
        raise ValueError("entities must be a list")
    seen = set()
    for e in entities:
        s, t, label = e.get("start"), e.get("end"), e.get("label")
        if type(s) is not int or type(t) is not int or not 0 <= s < t <= len(text):
            raise ValueError("invalid character interval")
        if label not in labels:
            raise ValueError("entity label not in annotated_labels")
        if e.get("text") != text[s:t]:
            raise ValueError("entity.text does not equal original text slice")
        if e.get("privacy", "unknown") not in PRIVACY_VALUES:
            raise ValueError("invalid privacy")
        key = (s, t, label)
        if key in seen:
            raise ValueError("duplicate annotation")
        seen.add(key)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out = []
    with Path(path).open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                validate_record(row)
            except (ValueError, TypeError, KeyError) as exc:
                # No input text in exception messages (training logs may be collected).
                raise ValueError(f"invalid record at {Path(path).name}:{line_no}") from exc
            out.append(row)
    return out


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            validate_record(row)
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def corpus_audit(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Check split identity/template/text leakage and report actual annotation coverage."""
    from collections import Counter
    summary: dict[str, Any] = {}
    texts: dict[str, str] = {}
    templates: dict[str, str] = {}
    identities: dict[str, str] = {}
    ids = set()
    for split, rows in splits.items():
        counts, privacy, families, negatives = Counter(), Counter(), set(), 0
        for row in rows:
            validate_record(row)
            if row["id"] in ids:
                raise ValueError("duplicate record id")
            ids.add(row["id"])
            key = hashlib.sha256(row["text"].casefold().encode()).hexdigest()
            if key in texts:
                raise ValueError("duplicate text within/across splits")
            texts[key] = split
            family = row.get("template_family")
            if family:
                if family in templates and templates[family] != split:
                    raise ValueError("template family leaked across splits")
                templates[family] = split
                families.add(family)
            for subject in row.get("synthetic_subjects", []):
                if subject in identities and identities[subject] != split:
                    raise ValueError("synthetic subject leaked across splits")
                identities[subject] = split
            main = [e for e in row["entities"] if e["label"] not in AUXILIARY]
            if not any(e.get("privacy") == "personal" for e in main):
                negatives += 1
            counts.update(e["label"] for e in row["entities"])
            privacy.update(e.get("privacy", "unknown") for e in main)
        summary[split] = {
            "records": len(rows), "entities": sum(counts.values()),
            "by_label": dict(sorted(counts.items())), "privacy_mentions": dict(privacy),
            "no_personal_pii_records": negatives, "template_families": len(families),
        }
    return {"schema_version": "1.0", "checks_passed": True, "splits": summary}
