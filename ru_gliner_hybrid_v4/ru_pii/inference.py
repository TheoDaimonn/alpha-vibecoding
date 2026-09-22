"""Local, synchronous Python inference wrapper for standard uni-encoder GLiNER.

No HTTP, no masking, no document logging, no automatic privacy declassification.
Real GLiNER integration needs an installed model; unit tests use an explicit fake.
"""
from __future__ import annotations
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Any, Protocol
import json
import math
from .schema import LABELS, AUXILIARY, Entity

@dataclass(frozen=True)
class TokenMap:
    tokens: tuple[str, ...]
    starts: tuple[int, ...]
    ends: tuple[int, ...]

@dataclass(frozen=True)
class Window:
    start: int
    end: int
    first_token: int
    last_token: int  # exclusive
    text: str

class Backend(Protocol):
    max_words: int
    max_width: int
    def tokenize(self, text: str, labels: Sequence[str]) -> TokenMap: ...
    def fits(self, tokens: Sequence[str], labels: Sequence[str]) -> bool: ...
    def predict(self, texts: list[str], labels: list[str], threshold: float, batch_size: int) -> list[list[dict[str, Any]]]: ...


def make_windows(text: str, labels: Sequence[str], backend: Backend, *, overlap: int = 32) -> Iterator[Window]:
    """Account for BOTH model splitter-token and prompt+subtoken budgets.

A bounded-width entity is contained in at least one window. A single splitter
word exceeding the actual context budget raises; it is never silently truncated.
"""
    mapping = backend.tokenize(text, labels)
    n = len(mapping.tokens)
    if n == 0:
        return
    if not (len(mapping.starts) == len(mapping.ends) == n):
        raise RuntimeError("invalid tokenizer alignment")
    for i,(start,end) in enumerate(zip(mapping.starts,mapping.ends)):
        if not 0 <= start < end <= len(text) or (i and start < mapping.ends[i-1]):
            raise RuntimeError("invalid or overlapping token offsets")
    overlap = max(overlap, backend.max_width-1)
    if not 0 <= overlap < backend.max_words:
        raise ValueError("overlap must be smaller than the word window")
    first = 0
    while first < n:
        lo, hi, best = first+1, min(n,first+backend.max_words), first
        while lo <= hi:
            mid=(lo+hi)//2
            if backend.fits(mapping.tokens[first:mid], labels):
                best=mid; lo=mid+1
            else:
                hi=mid-1
        if best == first:
            raise ValueError("one word plus label prompt exceeds model context")
        if best < n and best-first <= overlap:
            raise ValueError("context budget cannot fit overlap; shorten labels or input token")
        start,end = mapping.starts[first],mapping.ends[best-1]
        yield Window(start,end,first,best,text[start:end])
        if best == n:
            break
        first=best-overlap


class GLiNERBackend:
    """Version-bound adapter: gliner 0.2.29, standard uni-encoder span checkpoints."""
    def __init__(self, model: Any, *, subtoken_budget: int = 512, word_window: int = 256):
        if subtoken_budget < 64 or word_window < 16:
            raise ValueError("model budgets are too small")
        config=model.config
        if getattr(config, "labels_encoder", None) or getattr(config, "labels_decoder", None):
            raise ValueError("this adapter supports standard uni-encoder checkpoints only")
        self.model=model
        self.max_words=min(int(config.max_len),word_window)
        self.max_width=int(config.max_width)
        self.max_types=int(getattr(config,"max_types",30))
        self.tokenizer=model.data_processor.transformer_tokenizer
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("a fast transformer tokenizer is required for budget validation")
        native=int(getattr(self.tokenizer,"model_max_length",10**9))
        self.subtoken_budget=min(subtoken_budget,native if 0<native<10**7 else subtoken_budget)
        self.ent_token=str(config.ent_token)
        self.sep_token=str(config.sep_token)
        if hasattr(model,"configure_inference_packing"):
            model.configure_inference_packing(None)
        model.eval()

    def tokenize(self, text: str, labels: Sequence[str]) -> TokenMap:
        if not text.strip():
            return TokenMap((),(),())
        p=self.model.prepare_batch(text,list(labels))
        if not p["tokens"]:
            return TokenMap((),(),())
        ts=p["tokens"][0]
        starts,ends=p["start_token_map"][0],p["end_token_map"][0]
        return TokenMap(tuple(ts),tuple(int(starts[i]) for i in range(len(ts))),tuple(int(ends[i]) for i in range(len(ts))))

    def fits(self, tokens: Sequence[str], labels: Sequence[str]) -> bool:
        if len(tokens)>self.max_words:
            return False
        # Standard uni-encoder prompt construction. No manual text normalisation.
        words=[]
        for label in labels:
            words.extend([self.ent_token,label])
        words.append(self.sep_token)
        words.extend(tokens)
        encoded=self.tokenizer(words,is_split_into_words=True,add_special_tokens=True,
                               truncation=False,return_attention_mask=False)
        # Small explicit reserve; actual model tokenizer must still be integration-tested.
        return len(encoded["input_ids"])+8 <= self.subtoken_budget

    def predict(self, texts: list[str], labels: list[str], threshold: float, batch_size: int) -> list[list[dict[str, Any]]]:
        import torch
        # Recheck actual per-window splitter output (not just its original document slice).
        for text in texts:
            tm=self.tokenize(text,labels)
            if not self.fits(tm.tokens,labels):
                raise ValueError("window exceeds real tokenizer context")
        with torch.inference_mode():
            return self.model.inference(texts,labels,batch_size=batch_size,threshold=threshold,
                                        flat_ner=False,multi_label=True,packing_config=None)


class RussianPIIDetector:
    def __init__(self, backend: Backend, *, threshold: float = 0.5,
                 thresholds: dict[str,float] | None = None, batch_size: int = 8,
                 overlap: int = 32, max_chars: int = 2_000_000):
        if not 0 < threshold <= 1 or batch_size < 1 or max_chars < 1 or overlap < 0:
            raise ValueError("invalid inference settings")
        self.thresholds={k:threshold for k in LABELS}
        if thresholds:
            if not set(thresholds)<=LABELS.keys() or any(not math.isfinite(v) or not 0<v<=1 for v in thresholds.values()):
                raise ValueError("invalid per-type thresholds")
            self.thresholds.update(thresholds)
        self.backend=backend
        self.batch_size=batch_size
        self.overlap=overlap
        self.max_chars=max_chars
        self._lock=RLock()

    @classmethod
    def from_pretrained(cls, model_path: str | Path, *, device: str = "cpu",
                        allow_download: bool = False, revision: str | None = None,
                        subtoken_budget: int = 512, word_window: int = 256,
                        **kwargs: Any) -> "RussianPIIDetector":
        path=Path(model_path)
        if not path.is_dir() and not allow_download:
            raise FileNotFoundError("local model directory does not exist; remote loading is opt-in")
        from gliner import GLiNER
        load_kwargs={"local_files_only":not allow_download,"map_location":device,"load_tokenizer":True}
        if revision:
            load_kwargs["revision"]=revision
        model=GLiNER.from_pretrained(str(model_path),**load_kwargs)
        model.to(device)
        threshold_path=path/"thresholds.json"
        if threshold_path.is_file() and "thresholds" not in kwargs:
            saved=json.loads(threshold_path.read_text(encoding="utf-8"))
            kwargs["thresholds"]=saved.get("thresholds",saved)
        return cls(GLiNERBackend(model,subtoken_budget=subtoken_budget,word_window=word_window),**kwargs)

    def predict(self, text: str, *, labels: Sequence[str] | None = None,
                include_auxiliary: bool = True) -> list[Entity]:
        return self.predict_batch([text],labels=labels,include_auxiliary=include_auxiliary)[0]

    def predict_batch(self, texts: Sequence[str], *, labels: Sequence[str] | None = None,
                      include_auxiliary: bool = True) -> list[list[Entity]]:
        if isinstance(texts,str) or any(not isinstance(t,str) for t in texts):
            raise TypeError("texts must be a sequence of strings")
        if any(len(t)>self.max_chars for t in texts):
            raise ValueError("document exceeds configured character limit")
        codes=list(dict.fromkeys(labels if labels is not None else LABELS))
        if not codes or not set(codes)<=LABELS.keys():
            raise ValueError("unknown or empty canonical label set")
        prompts=[LABELS[k] for k in codes]
        reverse={LABELS[k]:k for k in codes}
        minimum=min(self.thresholds[k] for k in codes)
        results: list[dict[tuple[int,int,str],Entity]]=[{} for _ in texts]
        with self._lock:
            pending: list[tuple[int,Window]]=[]
            def flush() -> None:
                if not pending:
                    return
                raw=self.backend.predict([w.text for _,w in pending],prompts,minimum,self.batch_size)
                if len(raw)!=len(pending):
                    raise RuntimeError("model returned wrong batch length")
                for (doc,w),preds in zip(pending,raw):
                    for e in preds:
                        if e.get("label") not in reverse:
                            raise RuntimeError("model returned unknown entity label")
                        label=reverse[e["label"]]
                        score=float(e["score"])
                        if not math.isfinite(score) or not 0<=score<=1:
                            raise RuntimeError("invalid model confidence")
                        if score<self.thresholds[label]:
                            continue
                        s,t=e["start"],e["end"]
                        if type(s) is not int or type(t) is not int or not 0<=s<t<=len(w.text):
                            raise RuntimeError("model returned invalid offsets")
                        if e.get("text",w.text[s:t])!=w.text[s:t]:
                            raise RuntimeError("model changed the original entity text")
                        s,t=s+w.start,t+w.start
                        entity=Entity(s,t,label,score,texts[doc][s:t])
                        key=(s,t,label)
                        if key not in results[doc] or results[doc][key].score<score:
                            results[doc][key]=entity
                pending.clear()
            for doc,text in enumerate(texts):
                for window in make_windows(text,prompts,self.backend,overlap=self.overlap):
                    pending.append((doc,window))
                    if len(pending)>=self.batch_size:
                        flush()
            flush()
        output=[]
        for mapping in results:
            # A contextual subtype still implies the ordinary semantic entity.
            # This does NOT imply that declassification/masking is safe.
            for e in list(mapping.values()):
                base={"PUBLIC_PERSON":"PERSON","PUBLIC_ADDRESS":"ADDRESS"}.get(e.label)
                if base in codes and e.score>=self.thresholds[base]:
                    key=(e.start,e.end,base)
                    if key not in mapping:
                        mapping[key]=replace(e,label=base,source="gliner_context_subtype")
            public=[e for e in mapping.values() if e.label in AUXILIARY]
            entities=[]
            for e in mapping.values():
                if e.label in AUXILIARY and not include_auxiliary:
                    continue
                hint="public_candidate" if e.label in AUXILIARY or any(p.start<=e.start and e.end<=p.end for p in public) else "personal_or_unknown"
                entities.append(replace(e,privacy_hint=hint))
            output.append(sorted(entities,key=lambda e:(e.start,e.end,e.label)))
        return output
