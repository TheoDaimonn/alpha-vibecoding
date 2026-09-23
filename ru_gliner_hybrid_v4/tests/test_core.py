from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from ru_pii.schema import Entity, validate_record, TASK_REQUIRED_LABELS
from ru_pii.inference import TokenMap, RussianPIIDetector, make_windows
from ru_pii.prepare import prepare_records
from ru_pii.adapters import parse_serialized, align_tokens, bio_record, char_record, brat_records, SAFE_MAP, russian_fraction
from ru_pii.metrics import span_metrics, calibrate
from ru_pii.real_corpus import (
    parse_nerel_row,
    parse_factru_item,
    parse_factru_bio_item,
    parse_collection3_item,
    parse_jayguard_item,
    parse_redmadrobot_item,
    dedupe_splits,
    audit_real_corpus,
)
from ru_pii.benchmarking import BenchSpan, _merge_same_label, _score, project_prediction
from ru_pii.synthetic_supplement import make_splits as make_supplement_splits, audit as audit_supplement

ROOT = Path(__file__).resolve().parents[1]


class FakeBackend:
    """Small deterministic test double. Never exported as a trained model."""
    max_width = 12

    def __init__(self, max_words=64, budget=512):
        self.max_words = max_words
        self.budget = budget

    def tokenize(self, text, labels):
        matches = list(re.finditer(r"\w+|[^\w\s]", text))
        return TokenMap(
            tuple(m.group() for m in matches),
            tuple(m.start() for m in matches),
            tuple(m.end() for m in matches),
        )

    def fits(self, tokens, labels):
        return len(tokens) <= self.max_words and sum(max(1, len(t) // 4) for t in tokens) + 2 * len(labels) <= self.budget

    def predict(self, texts, labels, threshold, batch_size):
        answer = []
        for text in texts:
            entities = []
            for literal, label in [
                ("Иван", "person"),
                ("Тула", "city"),
                ("Тула, Мира, 7", "address"),
                ("Пушкин", "public person reference"),
                ("Пушкин", "person"),
            ]:
                if label in labels:
                    for match in re.finditer(re.escape(literal), text):
                        if 0.9 >= threshold:
                            entities.append({
                                "start": match.start(), "end": match.end(), "text": match.group(),
                                "label": label, "score": 0.9,
                            })
            answer.append(entities)
        return answer


def sample(text="Иван пришёл", split="train"):
    return {
        "id": "fixture-x",
        "text": text,
        "entities": [{"start": 0, "end": 4, "label": "PERSON", "text": "Иван", "privacy": "unknown"}],
        "annotated_labels": ["PERSON"],
        "split": split,
        "source": "unit-test-fixture",
        "synthetic": False,
    }


@pytest.mark.parametrize("field,value", [("start", -1), ("end", 999), ("text", "wrong"), ("label", "DATE"), ("privacy", "secret")])
def test_invalid_annotation(field, value):
    row = sample()
    row["entities"][0][field] = value
    with pytest.raises(ValueError):
        validate_record(row)


def test_duplicate_annotation_rejected():
    row = sample()
    row["entities"] *= 2
    with pytest.raises(ValueError):
        validate_record(row)


def test_empty_text_rejected():
    row = sample()
    row["text"] = " "
    with pytest.raises(ValueError):
        validate_record(row)


def test_entity_json():
    assert Entity(0, 4, "PERSON", 0.9, "Иван").to_dict()["text"] == "Иван"


def test_empty_inference():
    detector = RussianPIIDetector(FakeBackend())
    assert detector.predict_batch(["", "  "]) == [[], []]


def test_unicode_original_offsets():
    text = "🙂\n  Иван, готово."
    entity = RussianPIIDetector(FakeBackend()).predict(text, labels=["PERSON"])[0]
    assert text[entity.start:entity.end] == entity.text == "Иван"
    assert entity.start == 4


def test_nested_address_preserved():
    entities = RussianPIIDetector(FakeBackend()).predict("Тула, Мира, 7")
    assert {e.label for e in entities} == {"CITY", "ADDRESS"}


def test_public_hint_never_discards_person():
    entities = RussianPIIDetector(FakeBackend()).predict("Пушкин", include_auxiliary=False)
    assert len(entities) == 1 and entities[0].label == "PERSON"
    assert entities[0].privacy_hint == "public_candidate"


def test_deduplication_overlap():
    text = "слово " * 25 + "Иван " + "слово " * 60
    entities = RussianPIIDetector(FakeBackend(max_words=40), overlap=16).predict(text, labels=["PERSON"])
    assert len(entities) == 1 and entities[0].start == 150


def test_late_entity_100k_words():
    text = "слово " * 100000 + "Иван"
    entities = RussianPIIDetector(FakeBackend(max_words=256), overlap=16).predict(text, labels=["PERSON"])
    assert len(entities) == 1 and entities[0].end == len(text)


def test_window_coverage_and_overlap():
    text = " ".join(f"t{i}" for i in range(100))
    windows = list(make_windows(text, ["person"], FakeBackend(max_words=30), overlap=12))
    assert windows[0].first_token == 0 and windows[-1].last_token == 100
    assert all(a.last_token - b.first_token >= 11 for a, b in zip(windows, windows[1:]))


def test_single_oversized_token_fails():
    with pytest.raises(ValueError):
        list(make_windows("a" * 10000, ["person"], FakeBackend(), overlap=16))


def test_remote_load_is_opt_in():
    with pytest.raises(FileNotFoundError):
        RussianPIIDetector.from_pretrained("not-a-local-model")


def test_prepare_inclusive_end():
    row = sample("Иван Петров")
    row["entities"][0].update(end=11, text="Иван Петров")
    output, _ = prepare_records([row], FakeBackend())
    assert output[0]["ner"] == [[0, 1, "person"]]


def test_prepare_negative_explicit_labels():
    row = sample()
    row["entities"] = []
    output, _ = prepare_records([row], FakeBackend())
    assert output[0]["ner"] == [] and output[0]["ner_labels"] == ["person"]


@pytest.mark.parametrize("split", ["test", "challenge", "benchmark"])
def test_cannot_train_on_holdout(split):
    with pytest.raises(ValueError):
        prepare_records([sample(split=split)], FakeBackend())


def test_partial_positive_not_taught_as_negative():
    text = " ".join(["слово"] * 70)
    matches = list(re.finditer(r"\w+", text))
    start, end = matches[27].start(), matches[33].end()
    row = {
        "id": "partial", "text": text,
        "entities": [{"start": start, "end": end, "label": "PERSON", "text": text[start:end], "privacy": "unknown"}],
        "annotated_labels": ["PERSON", "EMAIL"], "split": "train", "source": "unit-test-fixture", "synthetic": False,
    }
    output, stats = prepare_records([row], FakeBackend(max_words=32), overlap=12)
    assert stats["partially_annotated_label_windows"] > 0
    assert any(x["ner"] for x in output)
    assert any(x["ner_labels"] == ["email"] for x in output)


@pytest.mark.parametrize("value", ['[1,2]', "[{'a': 1}]", [1, 2]])
def test_safe_serialized_parsing(value):
    assert isinstance(parse_serialized(value), list)


def test_no_eval_execution():
    with pytest.raises(ValueError):
        parse_serialized("__import__('os').system('echo UNSAFE')")


def test_russian_language_guard():
    assert russian_fraction("это русский текст") > 0.99
    assert russian_fraction("this is an English sentence") < 0.01


def test_monotonic_alignment_repeats():
    assert align_tokens("Иван  Иван", ["Иван", "Иван"]) == [(0, 4), (6, 10)]


def test_alignment_does_not_skip_characters():
    with pytest.raises(ValueError):
        align_tokens("Иван, Иван", ["Иван", "Иван"])


def test_bio_conversion():
    record = bio_record(
        {"text": "Иван Петров", "tokens": ["Иван", "Петров"], "ner_tags": ["B-PER", "I-PER"]},
        record_id="a", source="test", split="train", label_map={"PER": "PERSON"},
        annotated_source_labels=["PER"], license_="test",
    )
    assert record["entities"][0]["text"] == "Иван Петров"
    assert record["annotated_labels"] == ["PERSON"]


def test_generic_date_and_location_not_mapped_to_pii_roles():
    assert "DATE" not in SAFE_MAP and "LOC" not in SAFE_MAP and "NATIONALITY" not in SAFE_MAP


def test_char_partial_inventory():
    record = char_record(
        {"text": "Иван, 01.01.2020", "entities": [
            {"start": 0, "end": 4, "type": "PER", "text": "Иван"},
            {"start": 6, "end": 16, "type": "DATE", "text": "01.01.2020"},
        ]},
        record_id="a", source="x", split="train", text_field="text", span_field="entities",
        label_map={"PER": "PERSON"}, annotated_source_labels=["PER"], license_="x",
    )
    assert record["annotated_labels"] == ["PERSON"] and len(record["entities"]) == 1


def test_brat_discontinuous_rejected():
    with pytest.raises(ValueError):
        brat_records("Иван", "T1\tPER 0 2;3 4\tИв н", record_id="x", label_map={"PER": "PERSON"}, annotated_source_labels=["PER"], license_="x")


def test_exact_span_metrics():
    row = sample()
    metrics = span_metrics([row], [[{"start": 0, "end": 4, "label": "PERSON", "score": 0.9}]])
    assert metrics["exact_span_micro"]["f1"] == 1.0


def test_unknown_external_labels_not_counted_as_false_negative():
    row = sample()
    metrics = span_metrics([row], [[{"start": 0, "end": 4, "label": "EMAIL", "score": 0.9}]])
    assert "EMAIL" not in metrics["by_label"]


def test_no_test_threshold_selection():
    with pytest.raises(ValueError):
        calibrate([sample(split="test")], [[]])


def test_calibration_support_guard():
    result = calibrate([sample(split="dev")], [[]])
    assert result["thresholds"]["PERSON"] == 0.5


def test_archive_contains_focused_synthetic_supplement_but_not_synthetic_only_pipeline():
    assert (ROOT / "data/supplement/train.jsonl").is_file()
    assert (ROOT / "ru_pii/synthetic_supplement.py").is_file()
    assert (ROOT / "scripts/fetch_redmadrobot_train.py").is_file()
    assert (ROOT / "scripts/build_hybrid_corpus.py").is_file()


def test_registry_defaults_mix_sourced_data_and_focused_supplement():
    registry = json.loads((ROOT / "configs/datasets.json").read_text(encoding="utf-8"))
    defaults = {r["id"]: r for r in registry if r.get("default")}
    assert {"nerel", "factrueval2016", "redmadrobot_pii_train", "task_synthetic_supplement"} <= set(defaults)
    assert defaults["redmadrobot_pii_train"]["synthetic_text"] == "mixed"
    assert defaults["task_synthetic_supplement"]["synthetic_text"] is True
    assert "collection3" not in defaults  # license metadata requires separate review


def test_task_required_label_inventory_matches_spec_fields():
    expected = {
        "PERSON", "BIRTH_DATE", "BIRTH_PLACE", "PASSPORT_SERIES", "PASSPORT_NUMBER",
        "CITIZENSHIP", "PASSPORT_ISSUER", "PASSPORT_DIVISION_CODE", "PASSPORT_ISSUE_DATE",
        "DRIVER_LICENSE_SERIES", "DRIVER_LICENSE_NUMBER", "ADDRESS", "COUNTRY", "POSTAL_CODE",
        "CITY", "STREET", "HOUSE", "APARTMENT", "EMAIL", "PHONE", "INN", "CARD_NUMBER",
        "CVV", "PIN", "CARDHOLDER",
    }
    assert TASK_REQUIRED_LABELS == expected


def test_synthetic_supplement_covers_every_required_class():
    splits = make_supplement_splits(seed=7, sizes={"train": 110, "dev": 110, "test": 110})
    report = audit_supplement(splits)
    assert report["splits"]["train"]["missing_required"] == []


def test_benchmark_merge_person_parts_and_overlap():
    text = "Иван Иванов"
    merged = _merge_same_label([BenchSpan(0, 4, "PERSON"), BenchSpan(5, 11, "PERSON")], text)
    assert merged == [BenchSpan(0, 11, "PERSON")]
    assert _score([merged], [[BenchSpan(0, 11, "PERSON")]], overlap=False)["micro"]["f1"] == 1.0


def test_benchmark_projection():
    assert project_prediction(Entity(0, 4, "PASSPORT_NUMBER", 0.9, "1234")).label == "PASSPORT"
    assert project_prediction(Entity(0, 4, "BIRTH_DATE", 0.9, "2000")) is None


# The following are tiny parser fixtures only; they never enter data/real.
def test_nerel_birth_relations_are_role_specific():
    text = "Иван Иванов родился 1 января 1990 года в Москве."
    person, dob, city = "Иван Иванов", "1 января 1990 года", "Москве"
    ps, ds, cs = text.index(person), text.index(dob), text.index(city)
    row = {
        "id": 1, "text": text,
        "entities": [
            f"T1\tPERSON {ps} {ps + len(person)}\t{person}",
            f"T2\tDATE {ds} {ds + len(dob)}\t{dob}",
            f"T3\tCITY {cs} {cs + len(city)}\t{city}",
        ],
        "relations": ["R1\tDATE_OF_BIRTH Arg1:T1 Arg2:T2", "R2\tPLACE_OF_BIRTH Arg1:T1 Arg2:T3"],
    }
    out = parse_nerel_row(row, split="train")
    labels = {(e["label"], e["text"]) for e in out["entities"]}
    assert ("PERSON", person) in labels
    assert ("BIRTH_DATE", dob) in labels
    assert ("BIRTH_PLACE", city) in labels
    assert out["synthetic"] is False


def test_nerel_generic_date_does_not_become_birth_date():
    text = "Иван Иванов выступил 1 января 2020 года."
    person, date = "Иван Иванов", "1 января 2020 года"
    ps, ds = text.index(person), text.index(date)
    row = {"id": 2, "text": text, "entities": [
        f"T1\tPERSON {ps} {ps + len(person)}\t{person}",
        f"T2\tDATE {ds} {ds + len(date)}\t{date}",
    ], "relations": []}
    out = parse_nerel_row(row, split="train")
    assert "BIRTH_DATE" not in {e["label"] for e in out["entities"]}


def test_factru_real_bio_parser_maps_only_person():
    out = parse_factru_item({"id": "a", "tokens": ["Иван", "приехал", "в", "Москву"], "ner_tags": ["B-PER", "O", "O", "B-LOC"]}, split="train", index=0)
    assert [(e["label"], e["text"]) for e in out["entities"]] == [("PERSON", "Иван")]
    assert out["annotated_labels"] == ["PERSON"] and out["synthetic"] is False


def test_collection3_integer_tags_use_explicit_names():
    names = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
    out = parse_collection3_item({"id": "x", "tokens": ["Иван", "Петров", "в", "Москве"], "ner_tags": [1, 2, 0, 3]}, split="train", index=0, tag_names=names)
    assert [(e["label"], e["text"]) for e in out["entities"]] == [("PERSON", "Иван Петров")]
    assert out["synthetic"] is False


def test_factru_integer_tags_refuse_guessing_without_names():
    with pytest.raises(ValueError):
        parse_factru_bio_item({"tokens": ["Иван"], "ner_tags": [1]}, split="train", index=0)


def test_redmadrobot_parser_maps_task_and_bonus_pii():
    row = {
        "tokens": ["Иванов", "Иван", "паспорт", "45", "09", "123456", "ИНН", "123456789012"],
        "ner_tags": ["B-LAST_NAME", "B-FIRST_NAME", "O", "B-PASSPORT", "I-PASSPORT", "I-PASSPORT", "O", "B-INN"],
    }
    out = parse_redmadrobot_item(row, index=0)
    labels = {e["label"] for e in out["entities"]}
    assert {"PERSON", "PASSPORT", "INN"} <= labels
    assert out["source"] == "redmadrobot_pii_train"


def test_jayguard_is_optional_deidentified_context():
    out = parse_jayguard_item({
        "tokens": ["Пушкин", "жил", "на", "Тверской", "15"],
        "ner_tags": ["B-PUBLIC_PERSON", "O", "O", "B-STREET_ADDRESS", "I-STREET_ADDRESS"],
    }, split="train", index=0)
    assert {e["label"] for e in out["entities"]} == {"PUBLIC_PERSON", "ADDRESS"}
    assert out["synthetic"] is False


def test_real_corpus_dedupe_protects_heldout():
    train = parse_factru_item({"id": "a", "tokens": ["Иван"], "ner_tags": ["B-PER"]}, split="train", index=0)
    test = copy.deepcopy(train)
    test["id"], test["split"] = "factru-test-b", "test"
    out, dropped = dedupe_splits({"train": [train], "dev": [], "test": [test]})
    assert len(out["test"]) == 1 and out["train"] == [] and dropped["train"] == 1


def test_real_corpus_audit_marks_no_synthetic_rows():
    row = parse_factru_item({"id": "a", "tokens": ["Иван"], "ner_tags": ["B-PER"]}, split="train", index=0)
    assert audit_real_corpus({"train": [row]})["synthetic_rows"] == 0
