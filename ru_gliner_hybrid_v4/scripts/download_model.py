"""Materialise a complete PRETRAINED checkpoint for later offline fine-tuning.

This command downloads a base model. It does NOT fine-tune it and explicitly marks
its status so it cannot be confused with the output of scripts/train.py.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",default="gliner-community/gliner_small-v2.5")
    p.add_argument("--revision",default=None)
    p.add_argument("--output",default="models/base-small")
    args=p.parse_args()
    out=Path(args.output)
    if out.exists() and any(out.iterdir()):
        p.error("output must be empty to avoid confusing pretrained and fine-tuned weights")
    from huggingface_hub import HfApi
    from gliner import GLiNER
    revision=args.revision or HfApi().model_info(args.model).sha
    model=GLiNER.from_pretrained(args.model,revision=revision,load_tokenizer=True,map_location="cpu")
    model.save_pretrained(str(out),safe_serialization=True)
    manifest={"base_model":args.model,"revision":revision,"training_executed":False,
              "status":"downloaded_pretrained_only","created_at":datetime.now(timezone.utc).isoformat()}
    (out/"BASE_MODEL_MANIFEST.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("Complete pretrained checkpoint saved. No fine-tuning performed by this command.")

if __name__=="__main__":
    main()
