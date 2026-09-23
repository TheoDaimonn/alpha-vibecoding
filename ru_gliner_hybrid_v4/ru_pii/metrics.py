"""Exact typed-span metrics plus character-level privacy coverage. No claimed scores."""
from __future__ import annotations
from collections import Counter
from typing import Any
from .schema import AUXILIARY


def prf(tp:int,fp:int,fn:int)->dict[str,float|int]:
    p=tp/(tp+fp) if tp+fp else 0.0
    r=tp/(tp+fn) if tp+fn else 0.0
    return {"tp":tp,"fp":fp,"fn":fn,"precision":p,"recall":r,"f1":2*p*r/(p+r) if p+r else 0.0}


def span_metrics(rows:list[dict], predictions:list[list[dict]])->dict[str,Any]:
    if len(rows)!=len(predictions):
        raise ValueError("prediction/document count mismatch")
    counts: dict[str,Counter]={}
    char_tp=char_fp=char_fn=leaky=privacy_docs=0
    for row,pred in zip(rows,predictions):
        active=set(row["annotated_labels"])
        gold={(e["start"],e["end"],e["label"]) for e in row["entities"]}
        got=set()
        for e in pred:
            s,t=e["start"],e["end"]
            if type(s) is not int or type(t) is not int or not 0<=s<t<=len(row["text"]):
                raise ValueError("invalid prediction bounds")
            if e["label"] in active:
                got.add((s,t,e["label"]))
        for label in active:
            c=counts.setdefault(label,Counter())
            g={x for x in gold if x[2]==label}
            p={x for x in got if x[2]==label}
            c.update(tp=len(g&p),fp=len(p-g),fn=len(g-p))
        # Character privacy coverage is computed only for a separately adjudicated
        # privacy-evaluation set. Ordinary public NER corpora do not imply that a
        # named entity is private, so they must not create fake negatives.
        main=[e for e in row["entities"] if e["label"] not in AUXILIARY]
        if row.get("privacy_evaluation_eligible") is True and all(e.get("privacy") in {"personal","public"} for e in main):
            gchars={i for e in main if e["privacy"]=="personal" for i in range(e["start"],e["end"])}
            pchars={i for s,t,l in got if l not in AUXILIARY for i in range(s,t)}
            # Conservative policy: public candidates are NOT automatically removed.
            char_tp+=len(gchars&pchars); char_fn+=len(gchars-pchars); char_fp+=len(pchars-gchars)
            if gchars:
                privacy_docs+=1
                leaky+=bool(gchars-pchars)
    by_label={k:prf(c["tp"],c["fp"],c["fn"]) for k,c in sorted(counts.items())}
    total=Counter()
    for label,c in counts.items():
        if label not in AUXILIARY:
            total.update(c)
    return {"documents":len(rows),"exact_span_micro":prf(total["tp"],total["fp"],total["fn"]),"by_label":by_label,
            "privacy_character_conservative":prf(char_tp,char_fp,char_fn),
            "documents_with_personal_pii":privacy_docs,"documents_with_missed_personal_characters":leaky,
            "document_leakage_rate":leaky/privacy_docs if privacy_docs else None,
            "note":"Privacy-character metrics are populated only for separately adjudicated privacy-evaluation records. They are not the organisers' metric."}


def calibrate(rows:list[dict],predictions:list[list[dict]],*,beta:float=2.0,min_support:int=10)->dict[str,Any]:
    if any(r.get("split")!="dev" for r in rows):
        raise ValueError("threshold calibration is restricted to the dev split")
    if len(rows)!=len(predictions):
        raise ValueError("prediction/document count mismatch")
    labels=sorted({l for r in rows for l in r["annotated_labels"]})
    thresholds={}
    detail={}
    for label in labels:
        gold={(i,e["start"],e["end"]) for i,r in enumerate(rows) for e in r["entities"] if e["label"]==label}
        candidates=[(i,e) for i,p in enumerate(predictions) for e in p if e["label"]==label and label in rows[i]["annotated_labels"]]
        if len(gold)<min_support:
            thresholds[label]=0.5; detail[label]={"status":"insufficient_dev_support","support":len(gold)}
            continue
        best=None
        for j in range(1,20):
            t=j/20
            got={(i,e["start"],e["end"]) for i,e in candidates if e["score"]>=t}
            tp,fp,fn=len(got&gold),len(got-gold),len(gold-got)
            denom=(1+beta**2)*tp+beta**2*fn+fp
            f=(1+beta**2)*tp/denom if denom else 0.0
            candidate=(f,-t,t,tp,fp,fn)
            if best is None or candidate>best:
                best=candidate
        assert best is not None
        thresholds[label]=best[2]
        detail[label]={"status":"calibrated","support":len(gold),"f_beta":best[0],"beta":beta}
    return {"thresholds":thresholds,"selection_split":"dev","detail":detail}
