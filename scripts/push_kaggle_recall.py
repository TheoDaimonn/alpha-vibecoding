"""Build an isolated private Kaggle experiment. Pass --submit to upload and run."""
import argparse
import json
import re
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MODELS = {'tiny':'cointegrated/rubert-tiny2','base':'redmadrobot-rnd/rubert-base-pii-ner',
          'mdeberta':'microsoft/mdeberta-v3-base'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=MODELS,default='tiny')
    p.add_argument('--username',default='theodaimones888')
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--batch-size',type=int)
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--max-length',type=int,default=384)
    p.add_argument('--stride',type=int,default=96)
    p.add_argument('--extend-positions',action='store_true')
    p.add_argument('--gradient-checkpointing',action='store_true')
    p.add_argument('--gradient-accumulation-steps',type=int,default=1)
    p.add_argument('--run-suffix',default='v1')
    p.add_argument('--reuse-dataset',help='Attach an existing private dataset; upload code only')
    p.add_argument('--submit',action='store_true')
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',a.run_suffix):
        p.error('run-suffix must use lowercase letters, digits and single hyphens')
    if a.gradient_accumulation_steps < 1:
        p.error('gradient-accumulation-steps must be positive')
    if a.epochs < 1 or (a.batch_size is not None and a.batch_size < 1):
        p.error('epochs and batch-size must be positive')
    if not 0 <= a.stride < a.max_length - 2:
        p.error('stride must be smaller than max-length minus special tokens')
    folder=a.out or Path(tempfile.mkdtemp(prefix='pii-recall-'))
    folder.mkdir(parents=True,exist_ok=True)
    files=['experiments/pii_recall/train.py','ru_gliner_hybrid_v4/ru_pii/__init__.py',
           'ru_gliner_hybrid_v4/ru_pii/schema.py','ru_gliner_hybrid_v4/ru_pii/metrics.py']
    config={'model':MODELS[a.model],'epochs':a.epochs,
            'batch_size':a.batch_size or (32 if a.model=='tiny' else 8),
            'lr':a.lr,'max_length':a.max_length,'stride':a.stride,'fp16':True,
            'extend_positions':a.extend_positions,'gradient_checkpointing':a.gradient_checkpointing,
            'gradient_accumulation_steps':a.gradient_accumulation_steps}
    slug=f'ru-pii-recall-{a.model}-{a.run_suffix}'
    dataset_id=a.reuse_dataset or f'{a.username}/{slug}-bundle'
    runner=(ROOT/'scripts/kaggle_recall_train.py').read_text()
    if a.reuse_dataset:
        # Only source code and settings are sent; the corpus stays in its existing dataset.
        sources={name:(ROOT/name).read_text() for name in files}
        runner='RECALL_CONFIG = '+repr(config)+'\nRECALL_SOURCES = '+repr(sources)+'\n'+runner
    else:
        dataset=folder/'dataset'; dataset.mkdir(parents=True,exist_ok=True)
        files += [f'ru_gliner_hybrid_v4/data/hybrid/{split}.jsonl' for split in ('train','dev','test')]
        for name in files:
            dest=dataset/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,dest)
        (dataset/'experiment-config.json').write_text(json.dumps(config,indent=2))
        (dataset/'dataset-metadata.json').write_text(json.dumps({'id':dataset_id,'title':f'{slug}-bundle',
                                                                'licenses':[{'name':'other'}]},indent=2))
    (folder/'kaggle_recall_train.py').write_text(runner)
    (folder/'kernel-metadata.json').write_text(json.dumps({'id':f'{a.username}/{slug}','title':slug,
        'code_file':'kaggle_recall_train.py','language':'python','kernel_type':'script','is_private':True,
        'enable_gpu':True,'enable_internet':True,'dataset_sources':[dataset_id],
        'competition_sources':[],'kernel_sources':[],'model_sources':[]},indent=2))
    print(f'Bundle: {folder}',flush=True)
    if a.submit:
        kaggle=shutil.which('kaggle') or str(ROOT/'.venv/bin/kaggle')
        # Unique experiments: fail on an existing dataset rather than overwrite it silently.
        if not a.reuse_dataset:
            subprocess.run([kaggle,'datasets','create','-p',str(dataset),'-r','zip','-q'],check=True)
        subprocess.run([kaggle,'kernels','push','-p',str(folder)],check=True)
        print(f'https://www.kaggle.com/code/{a.username}/{slug}',flush=True)

if __name__=='__main__': main()
