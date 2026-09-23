"""Kaggle runner; settings live in the uploaded experiment-config.json."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

roots = list(Path('/kaggle/input').rglob('experiment-config.json'))
if len(roots) != 1:
    raise RuntimeError('Expected exactly one recall experiment bundle')
root = roots[0].parent
corpus_root = root
config = globals().get('RECALL_CONFIG', json.loads(roots[0].read_text()))
if 'RECALL_SOURCES' in globals():
    root = Path('/kaggle/working/recall-code')
    for name, source in RECALL_SOURCES.items():
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source)

subprocess.run([sys.executable,'-m','pip','install','-q','transformers==4.57.6','sentencepiece>=0.2,<0.3','protobuf'],check=True)
import torch
if not torch.cuda.is_available():
    raise RuntimeError('GPU was not assigned to this Kaggle run')
print(json.dumps({'gpu':torch.cuda.get_device_name(0),
                  'capability':torch.cuda.get_device_capability(0),
                  'training_precision':'fp16 mixed' if config.get('fp16') else 'fp32',
                  'epochs':config['epochs']}),flush=True)
out = Path('/kaggle/working/recall-model')
command = [sys.executable,str(root/'experiments/pii_recall/train.py'),
           '--model',config['model'],'--corpus',str(corpus_root/'ru_gliner_hybrid_v4/data/hybrid'),
           '--out',str(out),'--device','cuda','--epochs',str(config['epochs']),
           '--batch-size',str(config['batch_size']),
           '--lr',str(config['lr']),'--max-length',str(config['max_length']),
           '--stride',str(config['stride'])]
command += ['--gradient-accumulation-steps',str(config.get('gradient_accumulation_steps',1))]
for flag in ('fp16','extend_positions','gradient_checkpointing'):
    if config.get(flag):
        command.append('--'+flag.replace('_','-'))
# A single GPU is used intentionally; do not claim multi-GPU acceleration.
subprocess.run(command,check=True,env={**os.environ,'TOKENIZERS_PARALLELISM':'false'})
with tarfile.open('/kaggle/working/recall-model.tar.gz','w:gz') as archive:
    archive.add(out,arcname='recall-model')
print('Completed: recall-model.tar.gz; inspect test_metrics.json before deployment.',flush=True)
