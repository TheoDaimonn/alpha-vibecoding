"""Build a Russian NER corpus from public, non-generated source texts.

The module intentionally does NOT fabricate PII values or synthesize text.  It
converts existing annotations into the project's canonical character-span
schema while preserving partial-annotation semantics per source.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import re

from .schema import validate_record, write_jsonl
from .adapters import parse_serialized


@dataclass(frozen=True)
class SourceInfo:
    name: str
    url: str
    provenance: str
    annotation: str
    license_note: str
    use: str


SOURCES: dict[str, SourceInfo] = {
    "nerel": SourceInfo(
        name="NEREL",
        url="https://huggingface.co/datasets/iluvvatar/NEREL",
        provenance="real Russian Wikinews articles",
        annotation="manual nested entities + relations",
        license_note="source Wikinews text CC BY 2.5; retain NEREL attribution and verify annotation terms for your deployment",
        use="train/dev/test",
    ),
    "factrueval": SourceInfo(
        name="FactRuEval-2016",
        url="https://github.com/dialogue-evaluation/factRuEval-2016",
        provenance="real Russian news texts",
        annotation="manual competition NER",
        license_note="Kaggle mirror constantinwerner/multilingual-ner-dataset is CC0; preserve original FactRuEval attribution",
        use="train/dev/test",
    ),
    "collection3": SourceInfo(
        name="Collection3 / Persons-1000 extended",
        url="https://huggingface.co/datasets/RCC-MSU/collection3",
        provenance="real Russian news texts",
        annotation="manual PER/LOC/ORG NER",
        license_note="Hugging Face metadata is 'other'; use for private training only unless redistribution terms are reviewed",
        use="train/dev/test",
    ),
    "redmadrobot": SourceInfo(
        name="RedMadRobot Russian PII NER Training Dataset",
        url="https://huggingface.co/datasets/redmadrobot-rnd/pii_train",
        provenance="Russian production-log contexts with PII replaced before release, plus synthetic document-style rows and hard negatives",
        annotation="BIO PII annotations; 21 source entity types",
        license_note="MIT according to the Hugging Face dataset card",
        use="train only; paired pii_benchmark remains held out",
    ),
    "jayguard": SourceInfo(
        name="Jay Guard NER Benchmark",
        url="https://huggingface.co/datasets/just-ai/jayguard-ner-benchmark",
        provenance="real-world conversational contexts; source PII removed/replaced before public release",
        annotation="PII NER annotations",
        license_note="HF metadata says MIT while card body says Apache-2.0; resolve this metadata inconsistency before redistribution",
        use="optional training only; do not then report JayGuard as held-out benchmark",
    ),
}

# NEREL entity labels with an unambiguous canonical meaning for this task.
NEREL_ENTITY_MAP = {
    "PERSON": "PERSON",
    "COUNTRY": "COUNTRY",
    "CITY": "CITY",
    "STATE_OR_PROVINCE": "REGION",
    "STATE_OR_PROV": "REGION",
    "DISTRICT": "DISTRICT",
}

# Relation-derived labels are the key reason NEREL is useful: generic DATE and
# LOCATION mentions are NOT blindly relabelled as personal attributes.
NEREL_RELATION_MAP = {
    "DATE_OF_BIRTH": "BIRTH_DATE",
    "PLACE_OF_BIRTH": "BIRTH_PLACE",
}

FACTRU_MAP = {
    "PER": "PERSON",
    "PERSON": "PERSON",
}

COLLECTION3_MAP = {
    "PER": "PERSON",
    "PERSON": "PERSON",
}

# High-value Russian PII corpus.  Names are mapped to PERSON components; adjacent
# components are merged below. Coarse document labels are kept as coarse spans,
# while the synthetic supplement teaches task-specific series/number subspans.
REDMADROBOT_MAP = {
    "FIRST_NAME": "PERSON", "LAST_NAME": "PERSON", "MIDDLE_NAME": "PERSON",
    "COUNTRY": "COUNTRY", "REGION": "REGION", "DISTRICT": "DISTRICT",
    "CITY": "CITY", "STREET": "STREET", "HOUSE": "HOUSE",
    "EMAIL": "EMAIL", "PHONE": "PHONE",
    "PASSPORT": "PASSPORT", "INN": "INN", "CREDIT_CARD": "CARD_NUMBER",
    "DRIVER_LICENSE": "DRIVER_LICENSE",
    "SNILS": "SNILS", "OMS": "OMS", "MILITARY_ID": "MILITARY_ID",
    "BIRTH_CERTIFICATE": "BIRTH_CERTIFICATE", "IP_ADDRESS": "IP_ADDRESS", "URL": "URL",
}

JAYGUARD_MAP = {
    "PERSON": "PERSON",
    "PUBLIC_PERSON": "PUBLIC_PERSON",
    "STREET_ADDRESS": "ADDRESS",
}


_ENTITY_RE = re.compile(r"^(T\d+)\t([^\s]+)\s+(\d+)\s+(\d+)\t(.*)$", re.S)
_DISCONT_ENTITY_RE = re.compile(r"^(T\d+)\t([^\s]+)\s+((?:\d+\s+\d+;)+\d+\s+\d+)\t(.*)$", re.S)
_REL_RE = re.compile(r"^(R\d+)\t([^\s]+)\s+Arg1:(T\d+)\s+Arg2:(T\d+)\s*$")


def _entity(start: int, end: int, label: str, text: str, *, privacy: str = "unknown") -> dict[str, Any]:
    return {
        "start": start,
        "end": end,
        "label": label,
        "text": text[start:end],
        "privacy": privacy,
        "subject_id": None,
    }


def _base_record(*, record_id: str, text: str, split: str, source: str,
                 labels: Iterable[str], entities: list[dict[str, Any]],
                 source_url: str, provenance: str, annotation_method: str,
                 license_note: str) -> dict[str, Any]:
    record = {
        "id": record_id,
        "text": text,
        "split": split,
        "source": source,
        "annotated_labels": sorted(set(labels)),
        "entities": entities,
        "language": "ru",
        "synthetic": False,
        "provenance": provenance,
        "source_url": source_url,
        "annotation_method": annotation_method,
        "license": license_note,
    }
    validate_record(record)
    return record


def parse_nerel_row(row: dict[str, Any], *, split: str) -> dict[str, Any]:
    """Convert one official NEREL JSONL row.

    NEREL stores BRAT-like strings.  Birth date/place are derived ONLY from the
    explicit DATE_OF_BIRTH / PLACE_OF_BIRTH relations, never from generic dates
    or locations.  This avoids turning dates of events into personal data.
    """
    text = row["text"]
    ent_by_id: dict[str, tuple[str, int, int]] = {}
    canonical: list[dict[str, Any]] = []
    labels = set(NEREL_ENTITY_MAP.values()) | set(NEREL_RELATION_MAP.values())

    for raw in row.get("entities", []):
        if not isinstance(raw, str):
            raise ValueError("NEREL entity must be a BRAT string")
        m = _ENTITY_RE.match(raw)
        if m:
            eid, raw_type, s, e, stated = m.groups()
            s, e = int(s), int(e)
            if not 0 <= s < e <= len(text) or text[s:e] != stated:
                raise ValueError("NEREL character offsets do not match source text")
        else:
            discontinuous = _DISCONT_ENTITY_RE.match(raw)
            if not discontinuous:
                raise ValueError("unsupported NEREL entity syntax")
            eid, raw_type, spans, stated = discontinuous.groups()
            offsets = [int(value) for value in re.findall(r"\d+", spans)]
            pieces = list(zip(offsets[::2], offsets[1::2]))
            if not pieces or any(not 0 <= start < end <= len(text) for start, end in pieces):
                raise ValueError("NEREL character offsets do not match source text")
            s, e = pieces[0][0], pieces[-1][1]
        ent_by_id[eid] = (raw_type, s, e)
        mapped = NEREL_ENTITY_MAP.get(raw_type)
        if mapped:
            canonical.append(_entity(s, e, mapped, text, privacy="public" if mapped == "PERSON" else "unknown"))

    # Relation annotations point from PERSON/PROFESSION to DATE or place entity.
    for raw in row.get("relations", []):
        if not isinstance(raw, str):
            continue
        m = _REL_RE.match(raw)
        if not m:
            continue
        _, rtype, _arg1, arg2 = m.groups()
        label = NEREL_RELATION_MAP.get(rtype)
        target = ent_by_id.get(arg2)
        if not label or target is None:
            continue
        _, s, e = target
        canonical.append(_entity(s, e, label, text, privacy="public"))

    # Remove exact duplicate label/span pairs while preserving nested labels.
    uniq: dict[tuple[int, int, str], dict[str, Any]] = {}
    for e in canonical:
        uniq[(e["start"], e["end"], e["label"])] = e
    entities = sorted(uniq.values(), key=lambda x: (x["start"], x["end"], x["label"]))
    return _base_record(
        record_id=f"nerel-{split}-{row.get('id')}", text=text, split=split,
        source="nerel", labels=labels, entities=entities,
        source_url=SOURCES["nerel"].url, provenance=SOURCES["nerel"].provenance,
        annotation_method="manual_nested_ner_and_relation_to_char",
        license_note=SOURCES["nerel"].license_note,
    )


def _bio_to_record(tokens: list[str], tags: list[Any], *, record_id: str, split: str,
                   source: str, label_map: dict[str, str], labels: Iterable[str],
                   source_info: SourceInfo) -> dict[str, Any]:
    """Convert BIO tokens by reconstructing the published token stream with spaces.

    This never pretends to recover whitespace that a tokenized release discarded;
    the reconstructed string and offsets are internally exact and deterministic.
    """
    if len(tokens) != len(tags):
        raise ValueError("token/tag length mismatch")
    if any(not isinstance(t, str) for t in tokens):
        raise ValueError("tokens must be strings")
    if any(not isinstance(tag, str) for tag in tags):
        raise ValueError("integer BIO tags require an explicit source mapping; refusing to guess")

    starts: list[int] = []
    pieces: list[str] = []
    cursor = 0
    for i, tok in enumerate(tokens):
        if i:
            pieces.append(" ")
            cursor += 1
        starts.append(cursor)
        pieces.append(tok)
        cursor += len(tok)
    text = "".join(pieces)
    ends = [s + len(tok) for s, tok in zip(starts, tokens)]

    spans: list[dict[str, Any]] = []
    current: tuple[int, str] | None = None
    for i, tag in enumerate(tags + ["O"]):
        if tag == "O":
            prefix, raw = "O", None
        elif "-" in tag:
            prefix, raw = tag.split("-", 1)
            if prefix not in {"B", "I"}:
                raise ValueError("unsupported BIO prefix")
        else:
            raise ValueError("unsupported BIO tag")
        if prefix == "I" and (current is None or current[1] != raw):
            # Some FactRuEval rows contain an orphan I tag; treat it as the
            # start of a new span rather than discarding the annotated token.
            prefix = "B"
        if current is not None and (prefix != "I" or raw != current[1]):
            first, kind = current
            mapped = label_map.get(kind)
            if mapped:
                privacy = "public" if mapped == "PUBLIC_PERSON" else "unknown"
                spans.append(_entity(starts[first], ends[i - 1], mapped, text, privacy=privacy))
            current = None
        if prefix == "B":
            current = (i, raw)

    return _base_record(
        record_id=record_id, text=text, split=split, source=source,
        labels=labels, entities=spans, source_url=source_info.url,
        provenance=source_info.provenance,
        annotation_method="published_bio_to_reconstructed_char_offsets",
        license_note=source_info.license_note,
    )


def parse_factru_item(item: dict[str, Any], *, split: str, index: int) -> dict[str, Any]:
    # The HF mirror publishes token lists and string BIO tags.  LOC/ORG are not
    # mapped: a generic news location is not equivalent to a private address.
    tokens = parse_serialized(item["tokens"])
    # The mirror also includes an explicit string BIO field; prefer it over
    # integer ids whose label table is not retained in the JSON payload.
    tags = parse_serialized(item.get("ner_tags_str", item["ner_tags"]))
    if not isinstance(tokens, list) or not isinstance(tags, list):
        raise ValueError("FactRuEval tokens/ner_tags must be lists after safe parsing")
    return _bio_to_record(
        list(tokens), list(tags), record_id=f"factru-{split}-{item.get('id', index)}",
        split=split, source="factrueval2016", label_map=FACTRU_MAP,
        labels=sorted(set(FACTRU_MAP.values())), source_info=SOURCES["factrueval"],
    )



def parse_collection3_item(item: dict[str, Any], *, split: str, index: int, tag_names: list[str]) -> dict[str, Any]:
    """Convert a Collection3 sentence while keeping only PERSON supervision.

    LOC/ORG are deliberately ignored because a generic news location/organisation is
    not a private address or passport issuer.  `annotated_labels` therefore contains
    only PERSON and unobserved task labels are never treated as negatives.
    """
    tokens = list(item["tokens"])
    raw_tags = list(item["ner_tags"])
    tags = [tag_names[t] if isinstance(t, int) else str(t) for t in raw_tags]
    return _bio_to_record(
        tokens, tags, record_id=f"collection3-{split}-{item.get('id', index)}",
        split=split, source="collection3", label_map=COLLECTION3_MAP,
        labels=["PERSON"], source_info=SOURCES["collection3"],
    )

def parse_factru_bio_item(item: dict[str, Any], *, split: str, index: int, tag_names: list[str] | None = None) -> dict[str, Any]:
    """Convert a token/BIO FactRuEval row from Kaggle/HF-style tables."""
    tokens = parse_serialized(item["tokens"])
    raw_tags = parse_serialized(item["ner_tags"])
    if not isinstance(tokens, list) or not isinstance(raw_tags, list):
        raise ValueError("FactRuEval tokens/ner_tags must be lists")
    if raw_tags and isinstance(raw_tags[0], int):
        if not tag_names:
            raise ValueError("integer FactRuEval tags require tag_names")
        tags = [tag_names[t] for t in raw_tags]
    else:
        tags = [str(t) for t in raw_tags]
    return _bio_to_record(
        list(tokens), tags, record_id=f"factru-{split}-{item.get('id', index)}",
        split=split, source="factrueval2016", label_map=FACTRU_MAP,
        labels=["PERSON"], source_info=SOURCES["factrueval"],
    )

def parse_jayguard_item(item: dict[str, Any], *, split: str, index: int) -> dict[str, Any]:
    tokens = parse_serialized(item["tokens"])
    tags = parse_serialized(item["ner_tags"])
    if not isinstance(tokens, list) or not isinstance(tags, list):
        raise ValueError("JayGuard tokens/ner_tags must be lists after safe parsing")
    return _bio_to_record(
        list(tokens), list(tags), record_id=f"jayguard-{split}-{index}",
        split=split, source="jayguard", label_map=JAYGUARD_MAP,
        labels=sorted(set(JAYGUARD_MAP.values())), source_info=SOURCES["jayguard"],
    )



def _merge_adjacent_person_components(record: dict[str, Any]) -> dict[str, Any]:
    """Merge consecutive PERSON components separated only by whitespace.

    RedMadRobot publishes FIRST/LAST/MIDDLE names as separate BIO entities.  A
    whitespace-only merge recovers common full names without merging two people
    separated by punctuation or other tokens.
    """
    persons = sorted([e for e in record["entities"] if e["label"] == "PERSON"], key=lambda e: (e["start"], e["end"]))
    others = [e for e in record["entities"] if e["label"] != "PERSON"]
    merged: list[dict[str, Any]] = []
    for e in persons:
        if merged and record["text"][merged[-1]["end"]:e["start"]].isspace():
            prev = merged[-1]
            prev["end"] = e["end"]
            prev["text"] = record["text"][prev["start"]:prev["end"]]
        else:
            merged.append(dict(e))
    record = dict(record)
    record["entities"] = sorted(others + merged, key=lambda e: (e["start"], e["end"], e["label"]))
    validate_record(record)
    return record


def parse_redmadrobot_item(item: dict[str, Any], *, index: int, split: str = "train") -> dict[str, Any]:
    """Convert one pii_train CSV/HF row into the canonical schema."""
    tokens = parse_serialized(item["tokens"])
    tags = parse_serialized(item["ner_tags"])
    if not isinstance(tokens, list) or not isinstance(tags, list):
        raise ValueError("RedMadRobot tokens/ner_tags must be lists")
    record = _bio_to_record(
        list(tokens), [str(x) for x in tags],
        record_id=f"redmadrobot-{split}-{index}", split=split, source="redmadrobot_pii_train",
        label_map=REDMADROBOT_MAP, labels=sorted(set(REDMADROBOT_MAP.values())),
        source_info=SOURCES["redmadrobot"],
    )
    # Preserve the published text when it exactly matches the token stream apart
    # from ordinary spaces; otherwise retain deterministic reconstructed text.
    record["synthetic"] = None  # source is a documented mixture; row provenance is not published.
    record["provenance"] = SOURCES["redmadrobot"].provenance
    return _merge_adjacent_person_components(record)

def load_json_file(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def flatten_factru_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    if isinstance(payload, list):
        if len(payload) == 1 and isinstance(payload[0], dict) and isinstance(payload[0].get("data"), list):
            return payload[0]["data"]
        if all(isinstance(x, dict) and "tokens" in x for x in payload):
            return payload
    raise ValueError("unrecognised FactRuEval HF mirror JSON structure")


def _text_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.casefold().split()).encode("utf-8")).hexdigest()


def dedupe_splits(splits: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Remove exact normalized-text duplicates, protecting held-out splits first."""
    priority = ["test", "dev", "train"]
    seen: set[str] = set()
    dropped = Counter()
    out = {k: [] for k in splits}
    for split in priority:
        for row in splits.get(split, []):
            key = _text_hash(row["text"])
            if key in seen:
                dropped[split] += 1
                continue
            seen.add(key)
            out[split].append(row)
    for split in splits:
        out.setdefault(split, [])
    return out, dict(dropped)


def audit_real_corpus(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {"synthetic_rows": 0, "splits": {}}
    for split, rows in splits.items():
        by_source, by_label = Counter(), Counter()
        for row in rows:
            validate_record(row)
            if row.get("synthetic") is not False:
                result["synthetic_rows"] += 1
            by_source[row["source"]] += 1
            by_label.update(e["label"] for e in row["entities"])
        result["splits"][split] = {
            "records": len(rows),
            "entities": sum(by_label.values()),
            "by_source": dict(sorted(by_source.items())),
            "by_label": dict(sorted(by_label.items())),
        }
    return result


def write_corpus(output_dir: str | Path, splits: dict[str, list[dict[str, Any]]], *, manifest: dict[str, Any]) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    audit = audit_real_corpus(splits)
    if audit["synthetic_rows"]:
        raise ValueError("real-only corpus contains rows not explicitly marked synthetic=false")
    for split, rows in splits.items():
        write_jsonl(output / f"{split}.jsonl", rows)
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit
