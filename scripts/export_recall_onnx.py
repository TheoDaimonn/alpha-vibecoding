"""Export the trained recall RuBERT (encoder + per-type BIO head) to ONNX int8.

Loads the checkpoint produced by experiments/pii_recall/train.py (encoder/ +
head.pt), exports the full MultiBIO model to a dynamic-batch/seq ONNX graph and
quantizes it to int8 with onnxruntime dynamic quantization. Verifies argmax
agreement between PyTorch and the quantized ONNX session on a sample.

Usage:
    python scripts/export_recall_onnx.py \
        --model artifacts/recall-rubert-1024/model/recall-model \
        --out artifacts/recall-rubert-1024/onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ru_gliner_hybrid_v4"))
from ru_pii.schema import LABELS  # noqa: E402

TYPES = sorted(LABELS)


class MultiBIO(nn.Module):
    """Mirror of the training head so the exported graph matches training."""

    def __init__(self, encoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.0)
        self.head = nn.Linear(encoder.config.hidden_size, len(TYPES) * 3)

    def forward(self, ids, mask):
        states = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        return self.head(self.dropout(states)).reshape(*ids.shape, len(TYPES), 3)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True, help="checkpoint dir with encoder/ and head.pt")
    p.add_argument("--out", type=Path, required=True, help="output dir for model.onnx / model_int8.onnx")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model / "encoder", use_fast=True)
    encoder = AutoModel.from_pretrained(args.model / "encoder", local_files_only=True)
    model = MultiBIO(encoder)
    model.head.load_state_dict(torch.load(args.model / "head.pt", map_location="cpu", weights_only=True))
    model.eval()

    dummy = tokenizer("тестовая строка для экспорта onnx", return_tensors="pt")
    export_kwargs = dict(
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=args.opset,
    )
    onnx_path = args.out / "model.onnx"
    try:
        torch.onnx.export(
            model,
            (dummy["input_ids"], dummy["attention_mask"]),
            str(onnx_path),
            dynamo=False,
            **export_kwargs,
        )
    except TypeError:
        torch.onnx.export(model, (dummy["input_ids"], dummy["attention_mask"]), str(onnx_path), **export_kwargs)

    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path = args.out / "model_int8.onnx"
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)

    sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    logits_onnx = sess.run(
        None,
        {
            "input_ids": dummy["input_ids"].cpu().numpy(),
            "attention_mask": dummy["attention_mask"].cpu().numpy(),
        },
    )[0]
    with torch.no_grad():
        logits_pt = model(dummy["input_ids"], dummy["attention_mask"]).cpu().numpy()
    agreement = float((logits_onnx.argmax(-1) == logits_pt.argmax(-1)).mean())
    print(f"onnx int8 argmax agreement: {agreement:.4f}")
    print(f"wrote {onnx_path} and {int8_path}")


if __name__ == "__main__":
    main()