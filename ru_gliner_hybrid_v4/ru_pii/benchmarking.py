"""Benchmark helpers for Russian PII extraction.

The external benchmark adapters intentionally evaluate only the intersection of
benchmark labels and this project's task schema. Unsupported external labels are
reported, not silently counted as model false negatives.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from .adapters import align_tokens, parse_serialized


@dataclass(frozen=True)
class BenchSpan:
    start: int
    end: int
    label: str


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def _overlap(a: BenchSpan, b: BenchSpan) -> bool:
    return a.start < b.end and b.start < a.end


def _merge_same_label(spans: Iterable[BenchSpan], text: str, *, max_gap_chars: int = 3) -> list[BenchSpan]:
    """Merge same-label neighbours when the gap is punctuation/whitespace only.

    This is useful when one schema annotates an address/person as constituents
    while another emits a consolidated span. The raw exact metrics are also
    reported separately, so aggregation cannot hide boundary mismatch.
    """
    grouped: dict[str, list[BenchSpan]] = defaultdict(list)
    for span in spans:
        grouped[span.label].append(span)
    out: list[BenchSpan] = []
    for label, items in grouped.items():
        items = sorted(items, key=lambda x: (x.start, x.end))
        cur = items[0] if items else None
        for nxt in items[1:]:
            assert cur is not None
            overlap_or_touch = nxt.start <= cur.end
            gap = "" if overlap_or_touch else text[cur.end:nxt.start]
            can_merge = overlap_or_touch or (
                0 <= nxt.start - cur.end <= max_gap_chars
                and all(ch.isspace() or ch in ",.;:/\\-()№" for ch in gap)
            )
            if can_merge:
                cur = BenchSpan(cur.start, max(cur.end, nxt.end), label)
            else:
                out.append(cur)
                cur = nxt
        if cur is not None:
            out.append(cur)
    return sorted(out, key=lambda x: (x.start, x.end, x.label))


def _score(gold_docs: list[list[BenchSpan]], pred_docs: list[list[BenchSpan]], *, overlap: bool) -> dict[str, Any]:
    if len(gold_docs) != len(pred_docs):
        raise ValueError("gold/pred document count mismatch")
    total = Counter()
    per_label: dict[str, Counter] = defaultdict(Counter)
    for gold, pred in zip(gold_docs, pred_docs):
        if not overlap:
            gs = {(s.start, s.end, s.label) for s in gold}
            ps = {(s.start, s.end, s.label) for s in pred}
            labels = {x[2] for x in gs | ps}
            for label in labels:
                g = {x for x in gs if x[2] == label}
                p = {x for x in ps if x[2] == label}
                tp, fp, fn = len(g & p), len(p - g), len(g - p)
                per_label[label].update(tp=tp, fp=fp, fn=fn)
                total.update(tp=tp, fp=fp, fn=fn)
            continue

        # One-to-one greedy overlap matching by same class, preferring maximum IoU.
        used_gold: set[int] = set()
        used_pred: set[int] = set()
        candidates: list[tuple[float, int, int]] = []
        for pi, p in enumerate(pred):
            for gi, g in enumerate(gold):
                if p.label != g.label or not _overlap(p, g):
                    continue
                inter = min(p.end, g.end) - max(p.start, g.start)
                union = max(p.end, g.end) - min(p.start, g.start)
                candidates.append((inter / union, pi, gi))
        for _, pi, gi in sorted(candidates, reverse=True):
            if pi in used_pred or gi in used_gold:
                continue
            used_pred.add(pi)
            used_gold.add(gi)
        labels = {s.label for s in gold + pred}
        for label in labels:
            gp = sum(1 for gi, g in enumerate(gold) if g.label == label and gi in used_gold)
            total_g = sum(g.label == label for g in gold)
            total_p = sum(p.label == label for p in pred)
            tp, fp, fn = gp, total_p - gp, total_g - gp
            per_label[label].update(tp=tp, fp=fp, fn=fn)
            total.update(tp=tp, fp=fp, fn=fn)
    return {
        "micro": _prf(total["tp"], total["fp"], total["fn"]),
        "by_label": {k: _prf(v["tp"], v["fp"], v["fn"]) for k, v in sorted(per_label.items())},
    }


# Hivetrace types that overlap the hackathon requirements/model schema.
HIVETRACE_GOLD_MAP = {
    "NAME": "PERSON",
    "PHONE_NUMBER": "PHONE",
    "EMAIL": "EMAIL",
    "ADDRESS": "LOCATION",
    "BANK_CARD_NUMBER": "CARD_NUMBER",
    "CVC": "CVV",
    "INN": "INN",
    "PASSPORT_NUMBER": "PASSPORT",
}
HIVETRACE_GOLD_MAP["SNILS"] = "SNILS"
HIVETRACE_UNSUPPORTED = {"KPP", "OGRN", "OGRNIP", "TOKEN"}

# RedMadRobot coarse/common mapping. The benchmark itself recommends a coarse
# schema for cross-model comparison because fine-grained taxonomies differ.
REDMAD_GOLD_MAP = {
    "FIRST_NAME": "PERSON", "LAST_NAME": "PERSON", "MIDDLE_NAME": "PERSON",
    "COUNTRY": "LOCATION", "REGION": "LOCATION", "DISTRICT": "LOCATION",
    "CITY": "LOCATION", "STREET": "LOCATION", "HOUSE": "LOCATION",
    "EMAIL": "EMAIL", "PHONE": "PHONE",
    "PASSPORT": "PASSPORT", "INN": "INN", "CREDIT_CARD": "CARD_NUMBER",
    "DRIVER_LICENSE": "DRIVER_LICENSE",
    "SNILS": "SNILS", "OMS": "OMS", "MILITARY_ID": "MILITARY_ID",
    "BIRTH_CERTIFICATE": "BIRTH_CERTIFICATE", "URL": "URL", "IP_ADDRESS": "IP_ADDRESS",
}
REDMAD_UNSUPPORTED = set()


def project_prediction(entity: Any) -> BenchSpan | None:
    label = entity.label if hasattr(entity, "label") else entity["label"]
    start = int(entity.start if hasattr(entity, "start") else entity["start"])
    end = int(entity.end if hasattr(entity, "end") else entity["end"])
    mapped = {
        "PERSON": "PERSON",
        "ADDRESS": "LOCATION", "COUNTRY": "LOCATION", "POSTAL_CODE": "LOCATION",
        "REGION": "LOCATION", "CITY": "LOCATION", "STREET": "LOCATION",
        "HOUSE": "LOCATION", "BUILDING": "LOCATION", "APARTMENT": "LOCATION",
        "EMAIL": "EMAIL", "PHONE": "PHONE", "INN": "INN", "CARD_NUMBER": "CARD_NUMBER",
        "CVV": "CVV",
        "PASSPORT": "PASSPORT", "PASSPORT_SERIES": "PASSPORT", "PASSPORT_NUMBER": "PASSPORT",
        "DRIVER_LICENSE": "DRIVER_LICENSE", "DRIVER_LICENSE_SERIES": "DRIVER_LICENSE", "DRIVER_LICENSE_NUMBER": "DRIVER_LICENSE",
        "OTHER_ID": "OTHER_ID", "SNILS": "SNILS", "OMS": "OMS",
        "MILITARY_ID": "MILITARY_ID", "BIRTH_CERTIFICATE": "BIRTH_CERTIFICATE",
        "URL": "URL", "IP_ADDRESS": "IP_ADDRESS",
    }.get(label)
    return BenchSpan(start, end, mapped) if mapped else None


def _bio_gold(row: dict[str, Any], label_map: dict[str, str]) -> tuple[str, list[BenchSpan], set[str]]:
    tokens = parse_serialized(row["tokens"])
    tags = parse_serialized(row["ner_tags"])
    text = row.get("text") or " ".join(tokens)
    if not isinstance(tokens, list) or not isinstance(tags, list) or len(tokens) != len(tags):
        raise ValueError("invalid BIO row")
    offsets = align_tokens(text, tokens)
    gold: list[BenchSpan] = []
    unsupported: set[str] = set()
    current: tuple[int, str] | None = None
    for i, tag in enumerate(tags + ["O"]):
        if tag == "O":
            prefix, raw = "O", None
        else:
            prefix, raw = str(tag).split("-", 1)
        if current is not None and (prefix != "I" or raw != current[1]):
            first, kind = current
            if kind in label_map:
                gold.append(BenchSpan(offsets[first][0], offsets[i - 1][1], label_map[kind]))
            else:
                unsupported.add(kind)
            current = None
        if prefix == "B":
            current = (i, raw)
        elif prefix == "I" and (current is None or current[1] != raw):
            raise ValueError("invalid BIO sequence")
    return text, gold, unsupported


def evaluate_hivetrace(detector: Any, *, batch_size: int = 32, cache_dir: str | Path | None = None) -> dict[str, Any]:
    from datasets import load_dataset

    ds = load_dataset("hivetrace/pii-bench", cache_dir=str(cache_dir) if cache_dir else None)
    # Entity split gives exactly 70 examples per type. Restrict it to the task-overlap
    # types, avoiding unfair FN for labels that our model was never trained to emit.
    rows = [r for r in ds["entity"] if r["domain"] in HIVETRACE_GOLD_MAP]
    texts = [r["text"] for r in rows]
    preds_all: list[list[Any]] = []
    for i in range(0, len(texts), batch_size):
        preds_all.extend(detector.predict_batch(texts[i:i + batch_size], include_auxiliary=False))

    gold_docs: list[list[BenchSpan]] = []
    pred_docs: list[list[BenchSpan]] = []
    for row, preds in zip(rows, preds_all):
        gold = [BenchSpan(int(e["start"]), int(e["end"]), HIVETRACE_GOLD_MAP[e["type"]]) for e in row["entities"] if e["type"] in HIVETRACE_GOLD_MAP]
        projected = [x for x in (project_prediction(e) for e in preds) if x is not None]
        # Only labels actively scored for this benchmark subset; unrelated model
        # classes do not become false positives here.
        active = set(HIVETRACE_GOLD_MAP.values())
        projected = [x for x in projected if x.label in active]
        gold_docs.append(gold)
        pred_docs.append(projected)

    raw_exact = _score(gold_docs, pred_docs, overlap=False)
    merged_gold = [_merge_same_label(s, t) for s, t in zip(gold_docs, texts)]
    merged_pred = [_merge_same_label(s, t) for s, t in zip(pred_docs, texts)]
    merged_exact = _score(merged_gold, merged_pred, overlap=False)
    merged_overlap = _score(merged_gold, merged_pred, overlap=True)
    learned = {"PERSON", "LOCATION"}
    l_gold = [[s for s in doc if s.label in learned] for doc in merged_gold]
    l_pred = [[s for s in doc if s.label in learned] for doc in merged_pred]
    learned_overlap = _score(l_gold, l_pred, overlap=True)
    return {
        "benchmark": "hivetrace/pii-bench",
        "language": "ru",
        "split": "entity",
        "documents": len(rows),
        "scored_source_types": sorted(HIVETRACE_GOLD_MAP),
        "unsupported_source_types_excluded": sorted(HIVETRACE_UNSUPPORTED),
        "raw_exact": raw_exact,
        "merged_exact": merged_exact,
        "merged_overlap": merged_overlap,
        "name_address_overlap": learned_overlap,
        "notes": [
            "External held-out Russian benchmark; never used for training.",
            "ADDRESS is consolidated in gold, so raw exact and aggregated metrics are both reported.",
            "This is this project's transparent scorer, not a claim of bit-identical reproduction of the benchmark authors' harness.",
        ],
    }


def evaluate_redmadrobot(detector: Any, *, batch_size: int = 32, cache_dir: str | Path | None = None) -> dict[str, Any]:
    from datasets import load_dataset

    ds = load_dataset("redmadrobot-rnd/pii_benchmark", split="test", cache_dir=str(cache_dir) if cache_dir else None)
    rows = list(ds)
    texts: list[str] = []
    gold_docs: list[list[BenchSpan]] = []
    unsupported = Counter()
    for row in rows:
        text, gold, un = _bio_gold(row, REDMAD_GOLD_MAP)
        texts.append(text)
        gold_docs.append(gold)
        unsupported.update(un)

    preds_all: list[list[Any]] = []
    for i in range(0, len(texts), batch_size):
        preds_all.extend(detector.predict_batch(texts[i:i + batch_size], include_auxiliary=False))
    active = set(REDMAD_GOLD_MAP.values())
    pred_docs = [[x for x in (project_prediction(e) for e in preds) if x is not None and x.label in active] for preds in preds_all]
    merged_gold = [_merge_same_label(s, t) for s, t in zip(gold_docs, texts)]
    merged_pred = [_merge_same_label(s, t) for s, t in zip(pred_docs, texts)]
    common = {"PERSON", "LOCATION"}
    c_gold = [[s for s in doc if s.label in common] for doc in merged_gold]
    c_pred = [[s for s in doc if s.label in common] for doc in merged_pred]
    return {
        "benchmark": "redmadrobot-rnd/pii_benchmark",
        "language": "ru",
        "split": "test",
        "documents": len(rows),
        "scored_source_types": sorted(REDMAD_GOLD_MAP),
        "unsupported_source_types_excluded": dict(sorted(unsupported.items())),
        "merged_exact": _score(merged_gold, merged_pred, overlap=False),
        "merged_overlap": _score(merged_gold, merged_pred, overlap=True),
        "person_location_overlap": _score(c_gold, c_pred, overlap=True),
        "notes": [
            "External held-out Russian benchmark; never used for training.",
            "Fine-grained FIRST/LAST/MIDDLE and address components are projected to coarse PERSON/LOCATION before aggregation.",
            "This project scorer is intentionally transparent and may differ from the benchmark authors' official harness in edge cases.",
        ],
    }


def write_markdown_report(result: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Benchmark report", "", f"Model: `{result.get('model')}`", ""]
    for name, section in result.get("benchmarks", {}).items():
        lines += [f"## {name}", ""]
        if section.get("status") != "ok":
            lines += [f"Status: **{section.get('status')}**", "", f"Reason: `{section.get('reason', 'unknown')}`", ""]
            continue
        lines += [f"Documents: **{section.get('documents', 'n/a')}**", ""]
        for metric_name in ["exact_span_micro", "raw_exact", "merged_exact", "merged_overlap", "name_address_overlap", "person_location_overlap"]:
            metric = section.get(metric_name)
            if metric is None:
                continue
            micro = metric.get("micro", metric)
            if isinstance(micro, dict) and "f1" in micro:
                lines.append(f"- `{metric_name}`: P={micro['precision']:.4f}, R={micro['recall']:.4f}, F1={micro['f1']:.4f}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
