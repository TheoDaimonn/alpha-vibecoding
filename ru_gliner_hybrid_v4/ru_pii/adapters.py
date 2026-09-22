"""Conservative external-data converters. No generic DATE -> DOB or LOC -> ADDRESS."""
from __future__ import annotations
import ast
import json
import re
from typing import Any
from .schema import LABELS, validate_record

SAFE_MAP = {
    "NAME":"PERSON", "PERSON":"PERSON", "PER":"PERSON",
    "EMAIL":"EMAIL", "EMAIL_ADDRESS":"EMAIL",
    "PHONE":"PHONE", "PHONE_NUMBER":"PHONE", "TELEPHONENUM":"PHONE",
    "BANK_CARD_NUMBER":"CARD_NUMBER", "CREDITCARDNUMBER":"CARD_NUMBER", "CREDIT_CARD_NUMBER":"CARD_NUMBER",
    "CVV":"CVV", "CVC":"CVV", "CREDIT_CARD_SECURITY_CODE":"CVV",
    "PIN":"PIN", "INN":"INN", "TAXNUM":"INN",
    "ADDRESS":"ADDRESS", "STREET_ADDRESS":"ADDRESS",
    "COUNTRY":"COUNTRY", "CITY":"CITY", "STREET":"STREET",
    "HOUSE":"HOUSE", "BUILDINGNUM":"HOUSE", "ZIPCODE":"POSTAL_CODE", "POSTCODE":"POSTAL_CODE", "POSTAL_CODE":"POSTAL_CODE",
    "DOB":"BIRTH_DATE", "DATE_OF_BIRTH":"BIRTH_DATE",
    "DRIVERLICENSENUM":"DRIVER_LICENSE_NUMBER", "CARDHOLDER":"CARDHOLDER",
}
# Only these are exhaustively annotated in Hivetrace and unambiguously mapped here.
HIVETRACE_MAP = {k:SAFE_MAP[k] for k in ["NAME","PHONE_NUMBER","EMAIL","ADDRESS","BANK_CARD_NUMBER","CVC","INN"]}


def parse_serialized(value: Any, *, max_chars: int = 1_000_000) -> Any:
    if not isinstance(value,str):
        return value
    if len(value)>max_chars:
        raise ValueError("serialized annotation too large")
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        # ast.literal_eval, NEVER eval; length bound mitigates input abuse.
        try:
            return ast.literal_eval(value)
        except (SyntaxError,ValueError,MemoryError,RecursionError) as exc:
            raise ValueError("invalid serialized annotation") from exc


def russian_fraction(text: str) -> float:
    letters=[x for x in text if x.isalpha()]
    return sum("а"<=x.lower()<="я" or x.lower()=="ё" for x in letters)/max(1,len(letters))


def char_record(row: dict[str,Any], *, record_id: str, source: str, split: str,
                text_field: str, span_field: str, label_map: dict[str,str],
                annotated_source_labels: list[str], license_: str) -> dict[str,Any]:
    text=row[text_field]
    if not isinstance(text,str):
        raise ValueError("text field is not a string")
    spans=parse_serialized(row[span_field])
    if not isinstance(spans,list):
        raise ValueError("span field is not a list")
    active=sorted({label_map[k] for k in annotated_source_labels if k in label_map})
    entities=[]
    seen=set()
    for e in spans:
        raw=str(e.get("label",e.get("type","")))
        if raw not in label_map:
            continue
        if raw not in annotated_source_labels:
            raise ValueError("positive label absent from source annotation inventory")
        s,t=e.get("start"),e.get("end")
        if type(s) is not int or type(t) is not int or not 0<=s<t<=len(text):
            raise ValueError("invalid external offsets")
        stated=e.get("text",e.get("value",text[s:t]))
        if stated != text[s:t]:
            raise ValueError("external span value and original slice disagree")
        label=label_map[raw]
        key=(s,t,label)
        if key in seen:
            continue
        seen.add(key)
        entities.append(dict(start=s,end=t,label=label,text=text[s:t],privacy="unknown",subject_id=None))
    record=dict(id=record_id,text=text,entities=entities,annotated_labels=active,source=source,
                split=split,license=license_,synthetic=row.get("synthetic",None),
                annotation_method="external_char_offsets",language=row.get("language","unknown"))
    validate_record(record)
    return record


def align_tokens(text: str, tokens: list[str]) -> list[tuple[int,int]]:
    """Monotonic alignment, allowing only whitespace between tokens. No fuzzy fixes."""
    offsets=[]
    cursor=0
    for token in tokens:
        if not isinstance(token,str) or not token:
            raise ValueError("invalid token")
        while cursor<len(text) and text[cursor].isspace():
            cursor+=1
        if not text.startswith(token,cursor):
            raise ValueError("original text and token stream disagree")
        offsets.append((cursor,cursor+len(token)))
        cursor+=len(token)
    if text[cursor:].strip():
        raise ValueError("unrepresented non-whitespace text tail")
    return offsets


def bio_record(row: dict[str,Any], *, record_id: str, source: str, split: str,
               label_map: dict[str,str], annotated_source_labels: list[str],
               license_: str, tag_names: list[str] | None = None) -> dict[str,Any]:
    tokens=parse_serialized(row["tokens"])
    tags=parse_serialized(row["ner_tags"])
    if not isinstance(tokens,list) or not isinstance(tags,list) or len(tokens)!=len(tags):
        raise ValueError("BIO token/tag mismatch")
    text=row.get("text")
    reconstructed=text is None
    if reconstructed:
        text=" ".join(tokens)
    offsets=align_tokens(text,tokens)
    tags=[tag_names[t] if type(t) is int and tag_names is not None else t for t in tags]
    spans=[]
    current=None
    for i,tag in enumerate(tags+["O"]):
        if not isinstance(tag,str):
            raise ValueError("numeric BIO tags require source ClassLabel names")
        if tag=="O":
            prefix,kind="O",None
        elif "-" in tag:
            prefix,kind=tag.split("-",1)
            if prefix not in {"B","I"}:
                raise ValueError("only strict BIO is supported")
        else:
            raise ValueError("invalid BIO tag")
        if prefix=="I" and (current is None or current[1]!=kind):
            raise ValueError("orphan/mismatched I tag")
        if current is not None and (prefix!="I" or kind!=current[1]):
            start_token,raw=current
            s,t=offsets[start_token][0],offsets[i-1][1]
            spans.append(dict(start=s,end=t,type=raw,text=text[s:t]))
            current=None
        if prefix=="B":
            current=(i,kind)
    record=char_record({"text":text,"entities":spans},record_id=record_id,source=source,split=split,
                       text_field="text",span_field="entities",label_map=label_map,
                       annotated_source_labels=annotated_source_labels,license_=license_)
    record["annotation_method"]="bio_to_char"
    record["text_reconstructed_from_tokens"]=reconstructed
    return record


def brat_records(text: str, annotation: str, *, record_id: str, label_map: dict[str,str],
                 annotated_source_labels: list[str], license_: str, split: str="train") -> dict[str,Any]:
    spans=[]
    for line in annotation.splitlines():
        if not line.startswith("T"):
            continue
        fields=line.split("\t")
        if len(fields)!=3:
            raise ValueError("invalid BRAT text-bound annotation")
        descriptor=fields[1]
        if ";" in descriptor:
            raise ValueError("discontinuous BRAT spans require explicit task-specific handling")
        kind,start,end=descriptor.split()
        spans.append(dict(type=kind,start=int(start),end=int(end),text=fields[2]))
    return char_record({"text":text,"entities":spans},record_id=record_id,source="brat-import",split=split,
                       text_field="text",span_field="entities",label_map=label_map,
                       annotated_source_labels=annotated_source_labels,license_=license_)
