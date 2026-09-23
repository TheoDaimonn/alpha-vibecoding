"""Fetch a bounded, revision-pinned external sample; benchmark data never enters train.

Real network execution was not possible in the delivery environment. Schemas are
validated strictly at runtime; rejected rows are counted without printing text.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ru_pii.adapters import char_record, bio_record, russian_fraction, SAFE_MAP, HIVETRACE_MAP
from ru_pii.schema import write_jsonl, sha256_file

OPENPII_MAP={k:SAFE_MAP[k] for k in ["EMAIL","TELEPHONENUM","CITY","STREET","BUILDINGNUM","ZIPCODE","CREDITCARDNUMBER","DRIVERLICENSENUM"]}
# No FIRST_NAME->full-name merging, generic DATE->DOB, or full PASSPORT->six-digit number shortcut.
REDMAD_MAP={"EMAIL":"EMAIL","PHONE":"PHONE","PHONE_NUMBER":"PHONE", "INN":"INN",
            "CARD_NUMBER":"CARD_NUMBER","BANK_CARD_NUMBER":"CARD_NUMBER","CVV":"CVV", "PIN":"PIN",
            "COUNTRY":"COUNTRY","CITY":"CITY","STREET":"STREET","HOUSE":"HOUSE","APARTMENT":"APARTMENT"}


def check_role(item:dict,purpose:str,ack_review:bool=False)->None:
    if purpose=="train" and item["role"] in {"evaluation_only","evaluation_review"}:
        raise ValueError("benchmark data cannot be used for training")
    if item["license_state"]!="open_card" and not ack_review:
        raise ValueError("licence review is required; dataset is not enabled by default")
    if not item["hf_repo"]:
        raise ValueError("registry-only repository; use an audited local BRAT/CoNLL converter")


def is_russian_row(row:dict,text:str,item:dict,config:str|None=None)->bool:
    """Language metadata + Cyrillic sanity check, NOT a learned language detector.

    Known Russian corpora may legitimately contain bare IDs or Latin cardholders.
    A multilingual row without a language field is not assumed to be Russian.
    """
    if item.get("language_mode")=="ru" or (item.get("language_mode")=="ru subset" and config=="ru"):
        return True
    language=str(row.get("language",row.get("lang",""))).lower().replace("_","-")
    if language not in {"ru","rus","russian","русский"} and not language.startswith("ru-"):
        return False
    letters=sum(c.isalpha() for c in text)
    return letters<10 or russian_fraction(text)>=0.6


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--id",required=True)
    p.add_argument("--purpose",choices=["train","evaluation"],required=True)
    p.add_argument("--split",default=None)
    p.add_argument("--config",default=None,help="Hugging Face dataset configuration, e.g. ru")
    p.add_argument("--revision",default=None)
    p.add_argument("--max-records",type=int,default=2000)
    p.add_argument("--max-scan",type=int,default=50000)
    p.add_argument("--language",choices=["ru","any"],default="ru")
    p.add_argument("--mapping",help="JSON with label_map and exhaustively annotated source_labels")
    p.add_argument("--ack-license-review",action="store_true")
    p.add_argument("--output",default="data/external")
    args=p.parse_args()
    if args.max_records<1 or args.max_scan<args.max_records:
        p.error("invalid sample/scan limits")
    registry=json.loads(Path("configs/datasets.json").read_text(encoding="utf-8"))
    item=next((r for r in registry if r["id"]==args.id),None)
    if item is None:
        p.error("unknown registry id")
    check_role(item,args.purpose,args.ack_license_review)
    if item["adapter"] not in {"char","bio"}:
        p.error("schema-specific converter not implemented; registry entry is not an automatic importer")
    mapping=json.loads(Path(args.mapping).read_text()) if args.mapping else None
    if mapping is None:
        lm={"hivetrace":HIVETRACE_MAP,"redmadrobot":REDMAD_MAP,"openpii15m":OPENPII_MAP}.get(args.id)
        if lm is None:
            p.error("provide --mapping after reviewing this source's complete label inventory")
        mapping={"label_map":lm,"source_labels":list(lm)}
    from huggingface_hub import HfApi, hf_hub_download
    from datasets import load_dataset
    revision=args.revision or item.get("revision") or HfApi().dataset_info(item["hf_repo"]).sha
    split=args.split or ("domain" if args.id=="hivetrace" else "test" if args.id=="redmadrobot" else "train")
    if args.purpose=="train" and split.lower() in {"test","validation","dev","entity","domain"}:
        p.error("source split is held out; do not relabel it as train")
    if args.id=="redmadrobot":
        filename=hf_hub_download(item["hf_repo"],"test.csv",repo_type="dataset",revision=revision)
        stream=load_dataset("csv",data_files=filename,split="train",streaming=True)
    else:
        stream=load_dataset(item["hf_repo"],name=args.config,split=split,revision=revision,streaming=True)
    out=Path(args.output)/args.id
    out.mkdir(parents=True,exist_ok=True)
    counts=Counter()
    converted=[]
    target_split="train" if args.purpose=="train" else "benchmark"
    raw_path=out/f"{split}.raw.jsonl"
    temp=raw_path.with_suffix(".partial")
    tag_names=None
    if item["adapter"]=="bio":
        feature=getattr(stream,"features",None)
        if feature:
            seq=feature.get("ner_tags")
            tag_names=getattr(getattr(seq,"feature",None),"names",None)
    try:
        with temp.open("w",encoding="utf-8") as fh:
            for i,row in enumerate(stream):
                if i>=args.max_scan or len(converted)>=args.max_records:
                    break
                counts["scanned"]+=1
                text=row.get(item["text_field"],"") if item["text_field"] else " ".join(row.get("tokens",[]))
                if args.language=="ru" and (not isinstance(text,str) or not is_russian_row(row,text,item,args.config)):
                    counts["not_russian"]+=1
                    continue
                try:
                    common=dict(record_id=f"{args.id}:{revision}:{split}:{i}",source=item["hf_repo"],split=target_split,
                                label_map=mapping["label_map"],annotated_source_labels=mapping["source_labels"],
                                license_=str(item["declared_license"]))
                    if item["adapter"]=="char":
                        record=char_record(row,text_field=item["text_field"],span_field=item["span_field"],**common)
                    else:
                        record=bio_record(row,tag_names=tag_names,**common)
                    record["source_revision"]=revision
                    record["source_split"]=split
                    record["source_row"]=i
                    record["language"]="ru" if args.language=="ru" else row.get("language","unknown")
                    converted.append(record)
                    fh.write(json.dumps(row,ensure_ascii=False,default=str)+"\n")
                    counts["accepted"]+=1
                except (KeyError,ValueError,TypeError):
                    counts["invalid_or_incompatible_schema"]+=1
        manifest={"source":item,"revision":revision,"source_split":split,"purpose":args.purpose,
                  "requested_language":args.language,"counts":dict(counts),"mapping":mapping,
                  "status":"complete" if converted else "no_compatible_records","training_executed":False}
        (out/"fetch_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
        if not converted:
            raise RuntimeError("no compatible records; see aggregate fetch_manifest.json; no language fallback used")
        temp.replace(raw_path)
        canonical=out/f"{split}.canonical.jsonl"
        write_jsonl(canonical,converted)
        manifest["sha256"]={p.name:sha256_file(p) for p in [raw_path,canonical]}
        (out/"fetch_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps(dict(counts)))
    finally:
        if temp.exists():
            temp.unlink()

if __name__=="__main__":
    main()
