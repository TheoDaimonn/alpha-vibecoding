"""Export the local GLiNER checkpoint and compare full detector latency on CPU."""
import json
import time
from pathlib import Path
import sys
import shutil
import statistics
import importlib.metadata
import torch
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT), str(ROOT/'ru_gliner_hybrid_v4')]
from gliner import GLiNER
from ru_pii.inference import RussianPIIDetector, GLiNERBackend

SOURCE=ROOT/'ru_gliner_hybrid_v4/models/gliner-ru-pii-small'
DEST=ROOT/'artifacts/gliner-onnx'
REPORT=ROOT/'experiments/pii_recall/gliner_onnx_latency.json'
TEXTS=[f'Клиент Иванов Иван Иванович, email: user{i}@example.org, паспорт 4509 {i:06d}, телефон +7 900 123-45-67, ИНН 7707083893' for i in range(32)]

def measure(detector):
    for _ in range(3): detector.predict(TEXTS[0])
    measurements={}
    signatures=[]
    for batch in (1,8):
        samples=[]
        for i in range(20):
            texts=[TEXTS[(i+j)%len(TEXTS)] for j in range(batch)]
            start=time.perf_counter(); predictions=detector.predict_batch(texts)
            samples.append(time.perf_counter()-start)
            if batch==1:
                signatures.append([(e.start,e.end,e.label) for e in predictions[0]])
        ordered=sorted(samples)
        measurements[str(batch)]={'runs':len(samples),'p50_ms':statistics.median(samples)*1000,
            'p95_ms':ordered[int(.95*(len(ordered)-1))]*1000,
            'mean_ms':statistics.mean(samples)*1000,'texts_per_s':batch/statistics.mean(samples)}
    return measurements,signatures

def main():
    torch.set_num_threads(1)
    model=GLiNER.from_pretrained(str(SOURCE),local_files_only=True,map_location='cpu',load_tokenizer=True)
    model.eval()
    DEST.mkdir(parents=True,exist_ok=True)
    if not (DEST/'model.onnx').is_file():
        print('Exporting local GLiNER to ONNX',flush=True)
        print(model.export_to_onnx(DEST,quantize=False),flush=True)
    shutil.copy2(SOURCE/'thresholds.json',DEST/'thresholds.json')
    thresholds=json.loads((SOURCE/'thresholds.json').read_text()); thresholds=thresholds.get('thresholds',thresholds)
    report={'source':str(SOURCE.relative_to(ROOT)),'torch_threads':1,'onnx_threads':1,
        'scope':'CPU full detector including tokenization, labels, windows and postprocessing; no HTTP',
        'versions':{p:importlib.metadata.version(p) for p in ['gliner','torch','onnx','onnxruntime']}}
    reference=None
    for runtime in ['torch','onnxruntime']:
        print('Benchmarking '+runtime,flush=True)
        if runtime=='torch': backend_model=model
        else:
            options=ort.SessionOptions(); options.intra_op_num_threads=1; options.inter_op_num_threads=1
            backend_model=GLiNER.from_pretrained(str(DEST),local_files_only=True,map_location='cpu',load_tokenizer=True,runtime='onnxruntime',session_options=options)
        detector=RussianPIIDetector(GLiNERBackend(backend_model),thresholds=thresholds,batch_size=16)
        measurements,signatures=measure(detector)
        report[runtime]=measurements
        if reference is None: reference=signatures
        else: report['identical_span_predictions_out_of_20']=sum(a==b for a,b in zip(reference,signatures))
        REPORT.parent.mkdir(parents=True,exist_ok=True); REPORT.write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)

if __name__=='__main__': main()
