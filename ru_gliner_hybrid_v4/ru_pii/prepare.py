"""Convert canonical char annotations to actual GLiNER splitter-token annotations."""
from __future__ import annotations
from collections import Counter
from typing import Any
from .schema import LABELS, validate_record
from .inference import Backend, make_windows


def prepare_records(records: list[dict[str,Any]], backend: Backend, *, overlap: int = 32) -> tuple[list[dict[str,Any]],dict[str,Any]]:
    prepared=[]
    stats=Counter()
    for row in records:
        validate_record(row)
        if row.get("split") in {"test","challenge","benchmark"}:
            raise ValueError("held-out records are forbidden in training preparation")
        codes=row["annotated_labels"]
        max_types=getattr(backend,"max_types",30)
        if len(codes)>max_types:
            for start in range(0,len(codes),max_types):
                label_chunk=set(codes[start:start+max_types])
                chunk=dict(row)
                chunk["annotated_labels"]=sorted(label_chunk)
                chunk["entities"]=[e for e in row["entities"] if e["label"] in label_chunk]
                chunk_prepared,chunk_stats=prepare_records([chunk],backend,overlap=overlap)
                prepared.extend(chunk_prepared)
                stats.update(chunk_stats)
            continue
        prompts=[LABELS[k] for k in codes]
        tm=backend.tokenize(row["text"],prompts)
        starts={s:i for i,s in enumerate(tm.starts)}
        ends={e:i for i,e in enumerate(tm.ends)}
        entities=[
            e for e in row["entities"]
            if e["start"] in starts and e["end"] in ends
            and ends[e["end"]] - starts[e["start"]] + 1 <= backend.max_width
        ]
        stats["unalignable_annotations"] += len(row["entities"]) - len(entities)
        coverage=set()
        # Use an alternative contextual type instead of two positive classes for
        # exactly the same span. Some GLiNER processors historically collapsed
        # duplicate span keys. Canonical annotations retain BOTH semantic facts.
        contextual={}
        for i,e in enumerate(entities):
            target={"PERSON":"PUBLIC_PERSON","ADDRESS":"PUBLIC_ADDRESS"}.get(e["label"])
            if target:
                for j,aux in enumerate(entities):
                    if aux["label"]==target and (e["start"],e["end"])==(aux["start"],aux["end"]):
                        contextual[i]=j
                        break
        for i,e in enumerate(entities):
            if ends[e["end"]]-starts[e["start"]]+1>backend.max_width:
                raise ValueError(f"entity exceeds max_width in record {row['id']}; no silent span truncation")
        for w in make_windows(row["text"],prompts,backend,overlap=overlap):
            # A clipped positive must NOT be re-labelled as a negative.
            partial=set()
            contained=[]
            for i,e in enumerate(entities):
                intersects=e["start"]<w.end and w.start<e["end"]
                full=w.start<=e["start"] and e["end"]<=w.end
                if intersects and not full:
                    partial.add(e["label"])
                elif full:
                    contained.append((i,e))
            active=[k for k in codes if k not in partial]
            if not active:
                stats["empty_supervision_windows"]+=1
                continue
            local=backend.tokenize(w.text,[LABELS[k] for k in active])
            ls={s:i for i,s in enumerate(local.starts)}
            le={e:i for i,e in enumerate(local.ends)}
            ner=[]
            for i,e in contained:
                if i in contextual:
                    continue
                if e["label"] not in active:
                    continue
                s,t=e["start"]-w.start,e["end"]-w.start
                if s not in ls or t not in le:
                    raise ValueError("window changed entity-token alignment")
                ner.append([ls[s],le[t],LABELS[e["label"]]])
                coverage.add(i)
                coverage.update(original for original,auxiliary in contextual.items() if auxiliary==i)
            if not backend.fits(local.tokens,[LABELS[k] for k in active]):
                raise ValueError("prepared training example exceeds subtoken budget")
            prepared.append({"tokenized_text":list(local.tokens),"ner":ner,"ner_labels":[LABELS[k] for k in active]})
            stats["windows"]+=1
            stats["entity_targets"]+=len(ner)
            stats["partially_annotated_label_windows"]+=len(partial)
        missing = len(set(range(len(entities))) - coverage)
        stats["annotations_without_complete_window"] += missing
        stats["source_documents"]+=1
    return prepared,dict(stats)
