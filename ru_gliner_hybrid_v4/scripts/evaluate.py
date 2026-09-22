"""Evaluate an actual checkpoint; calibration only on dev, test only after selection."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ru_pii.schema import read_jsonl, sha256_file
from ru_pii.inference import RussianPIIDetector
from ru_pii.metrics import span_metrics, calibrate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",required=True)
    p.add_argument("--data",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--device",default="cpu")
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--calibrate",action="store_true")
    p.add_argument("--limit",type=int,default=None)
    p.add_argument("--save-predictions",action="store_true",help="saves offsets/labels/scores only, no entity text")
    args=p.parse_args()
    rows=read_jsonl(args.data)
    if args.limit is not None:
        rows=rows[:args.limit]
    if not rows:
        p.error("empty evaluation data")
    if args.calibrate and any(r.get("split")!="dev" for r in rows):
        p.error("calibration requires dev records")
    kwargs={"device":args.device,"batch_size":args.batch_size}
    if args.calibrate:
        kwargs.update(threshold=0.05,thresholds={})
    detector=RussianPIIDetector.from_pretrained(args.model,**kwargs)
    predictions=[]
    for start in range(0,len(rows),32):
        batch=rows[start:start+32]
        result=detector.predict_batch([r["text"] for r in batch])
        predictions.extend([[{k:v for k,v in e.to_dict().items() if k!="text"} for e in es] for es in result])
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    report=span_metrics(rows,predictions)
    report.update(model=args.model,data_sha256=sha256_file(args.data),subset_limit=args.limit)
    if args.calibrate:
        thresholds=calibrate(rows,predictions)
        Path(args.model,"thresholds.json").write_text(json.dumps(thresholds,ensure_ascii=False,indent=2),encoding="utf-8")
        report["thresholds_file"]=str(Path(args.model,"thresholds.json"))
        report["warning"]="Metrics above are raw low-threshold predictions; re-run without --calibrate for chosen-threshold metrics."
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    if args.save_predictions:
        path=output.with_suffix(".predictions.jsonl")
        path.write_text("".join(json.dumps({"id":r["id"],"predictions":p},ensure_ascii=False)+"\n" for r,p in zip(rows,predictions)),encoding="utf-8")
    print(json.dumps(report["exact_span_micro"]))

if __name__=="__main__":
    main()
