"""Compare local GLiNER FP32, dynamic INT8 and float8 weight-storage variants.

Float8 variants decode constant weights to FP32. They are not native FP8 compute.
Quality is a reproducible source-stratified dev subset, not the full dev set.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time
from collections import Counter

import ml_dtypes
import numpy as np
import onnx
from onnx import numpy_helper, helper, TensorProto
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ru_gliner_hybrid_v4')]
from gliner import GLiNER
from ru_pii.inference import RussianPIIDetector, GLiNERBackend
from ru_pii.schema import read_jsonl
from ru_pii.metrics import span_metrics

DIR=ROOT/'artifacts/gliner-onnx'
OUT=ROOT/'experiments/pii_recall/gliner_quantized'


def convert_float8(destination, dtype):
    model=onnx.load(str(DIR/'model.onnx'))
    prefix=[]; changed=0
    for initializer in model.graph.initializer:
        if initializer.data_type != TensorProto.FLOAT or len(initializer.dims)<2:
            continue
        arr=numpy_helper.to_array(initializer)
        if arr.size<1024: continue
        name=initializer.name
        scale=np.float32(max(float(np.max(np.abs(arr)))/float(ml_dtypes.finfo(dtype).max),1e-12))
        quantized=(arr/scale).astype(dtype)
        initializer.CopyFrom(numpy_helper.from_array(quantized,name+'__float8'))
        scale_tensor=numpy_helper.from_array(np.array(scale),name+'__scale')
        # Constant attributes avoid modifying the initializer list during iteration.
        prefix.extend([helper.make_node('Constant',[],[name+'__scale'],value=scale_tensor),
            helper.make_node('Cast',[name+'__float8'],[name+'__decoded'],to=TensorProto.FLOAT),
            helper.make_node('Mul',[name+'__decoded',name+'__scale'],[name])])
        changed+=1
    nodes=list(model.graph.node); del model.graph.node[:]; model.graph.node.extend(prefix+nodes)
    onnx.checker.check_model(model)
    onnx.save(model,str(destination))
    return changed


def subset():
    path=ROOT/'ru_gliner_hybrid_v4/data/hybrid/dev.jsonl'
    rows=read_jsonl(path); rng=random.Random(20260923)
    groups={source:[i for i,row in enumerate(rows) if row.get('source','unknown')==source]
            for source in sorted({r.get('source','unknown') for r in rows})}
    selected=[]
    for indices in groups.values():
        rng.shuffle(indices)
        selected.extend(indices[:round(384*len(indices)/len(rows))])
    selected=sorted(selected)
    return [rows[i] for i in selected],dict(indices=selected,seed=20260923,
        corpus_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        sources=dict(Counter(rows[i].get('source','unknown') for i in selected)))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--variant',choices=['fp32','int8','fp8','bf8'],required=True); args=parser.parse_args()
    torch.set_num_threads(1); OUT.mkdir(parents=True,exist_ok=True)
    filename={'fp32':'model.onnx','int8':'model_int8.onnx','fp8':'model_fp8_weights.onnx','bf8':'model_bf8_weights.onnx'}[args.variant]
    path=DIR/filename
    if not path.exists():
        print('Converting '+args.variant,flush=True)
        if args.variant=='int8':
            quantize_dynamic(str(DIR/'model.onnx'),str(path),weight_type=QuantType.QInt8,
                op_types_to_quantize=['MatMul','Gemm','Gather'],per_channel=False)
        else:
            changed=convert_float8(path,ml_dtypes.float8_e4m3fn if args.variant=='fp8' else ml_dtypes.float8_e5m2)
            print(f'Quantized {changed} matrix weights; FP32 computation',flush=True)
        gc.collect()
    options=ort.SessionOptions(); options.intra_op_num_threads=1; options.inter_op_num_threads=1
    print('Loading '+filename,flush=True)
    model=GLiNER.from_pretrained(str(DIR),local_files_only=True,map_location='cpu',load_tokenizer=True,
        runtime='onnxruntime',onnx_model_file=filename,session_options=options)
    thresholds=json.loads((DIR/'thresholds.json').read_text()); thresholds=thresholds.get('thresholds',thresholds)
    detector=RussianPIIDetector(GLiNERBackend(model),thresholds=thresholds,batch_size=8)
    rows,selection=subset()
    result={'variant':args.variant,'onnx_bytes':path.stat().st_size,'threads':1,
        'compute':'dynamic INT8 eligible operators; other operators FP32' if args.variant=='int8' else 'FP32',
        'selection':selection,'quality_scope':'source-stratified dev subset; exact typed spans, raw detector before API filtering',
        'latency_scope':'CPU full detector including tokenizer and decoding; batch=1, no HTTP',
        'versions':{'onnxruntime':ort.__version__,'onnx':onnx.__version__}}
    samples=[]
    for i in range(23):
        text=f'Клиент Иванов Иван Иванович, email: user{i}@example.org, паспорт 4509 {i:06d}, телефон +7 900 123-45-67, ИНН 7707083893'
        start=time.perf_counter(); detector.predict(text)
        if i>=3: samples.append(time.perf_counter()-start)
    result['latency_ms']={'p50':statistics.median(samples)*1000,'p95':sorted(samples)[18]*1000,'mean':statistics.mean(samples)*1000,'runs':20}
    print('Latency '+json.dumps(result['latency_ms']),flush=True)
    predictions=[]; started=time.perf_counter()
    for i in range(0,len(rows),8):
        predictions.extend([[e.to_dict() for e in entities] for entities in detector.predict_batch([r['text'] for r in rows[i:i+8]])])
        if i%32==0: print(f'Quality {i+8}/{len(rows)}',flush=True)
    result['quality_elapsed_s']=time.perf_counter()-started
    result['quality']=span_metrics(rows,predictions)
    result['by_source']={source:span_metrics([r for r in rows if r.get('source','unknown')==source],
        [p for r,p in zip(rows,predictions) if r.get('source','unknown')==source])['exact_span_micro'] for source in selection['sources']}
    (OUT/(args.variant+'.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(result['quality']['exact_span_micro']),flush=True)

if __name__=='__main__': main()
