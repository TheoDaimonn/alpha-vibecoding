"""Exact typed recall for an existing detector, with per-type support and P/R/F1."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ru_gliner_hybrid_v4')]
from ru_pii.schema import read_jsonl, TASK_REQUIRED_LABELS
from ru_pii.metrics import span_metrics

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--detector',choices=['student','transformer','rules'],default='student')
    p.add_argument('--split',choices=['dev','test'],default='dev')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    if a.detector=='student':
        from src.models.student_detector import StudentDetector
        detector=StudentDetector(ROOT/'artifacts/student-pii.pt')
    elif a.detector=='transformer':
        from src.models.transformer_detector import TransformerDetector
        detector=TransformerDetector(ROOT/'artifacts/transformer-pii.pt')
    else:
        from src.models.rules import RuleDetector
        detector=RuleDetector()
    rows=read_jsonl(ROOT/f'ru_gliner_hybrid_v4/data/hybrid/{a.split}.jsonl')
    predictions=[]; started=time.time()
    for i in range(0,len(rows),32):
        if i % 512 == 0: print(f'Evaluating {i}/{len(rows)}',flush=True)
        predictions.extend([[e.to_dict() for e in entities] for entities in detector.predict_batch([r['text'] for r in rows[i:i+32]])])
    result=span_metrics(rows,predictions)
    result['by_source']={}
    for source in sorted({r.get('source','unknown') for r in rows}):
        indices=[i for i,r in enumerate(rows) if r.get('source','unknown')==source]
        result['by_source'][source]=span_metrics([rows[i] for i in indices],[predictions[i] for i in indices])['exact_span_micro']
    result.update(detector=a.detector,split=a.split,elapsed_s=time.time()-started,
                  scope='Raw detector, before API public-context filtering; corpus annotation protocol, not organiser score')
    result['required_labels_below_095']=[l for l in sorted(TASK_REQUIRED_LABELS) if result['by_label'].get(l,{}).get('recall',0)<.95]
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result['exact_span_micro']),flush=True)
if __name__=='__main__': main()
