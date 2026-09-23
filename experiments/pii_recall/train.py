"""Independent per-type BIO heads: nested labels, partial supervision, full windows."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ru_gliner_hybrid_v4'))
from ru_pii.schema import LABELS, TASK_REQUIRED_LABELS, read_jsonl, sha256_file
from ru_pii.metrics import span_metrics

TYPES = sorted(LABELS)


def encode(row, tokenizer, max_length, stride):
    enc = tokenizer(row['text'], truncation=True, max_length=max_length,
                    stride=stride, return_overflowing_tokens=True, return_offsets_mapping=True)
    windows = []
    for ids, offsets in zip(enc['input_ids'], enc['offset_mapping']):
        targets = np.full((len(ids), len(TYPES)), -100, dtype=np.int64)
        valid = [i for i, (s, e) in enumerate(offsets) if e > s]
        for c, label in enumerate(TYPES):
            if label not in row['annotated_labels']:
                continue  # Unannotated != negative.
            targets[valid, c] = 0
            for ent in row['entities']:
                if ent['label'] != label:
                    continue
                indices = [i for i in valid if offsets[i][0] < ent['end'] and offsets[i][1] > ent['start']]
                for j, i in enumerate(indices):
                    targets[i, c] = 1 if j == 0 else 2
        windows.append({'input_ids': ids, 'offsets': offsets, 'targets': targets})
    return windows


def collate(items, pad_id):
    length = max(len(x['input_ids']) for x in items)
    ids = torch.full((len(items), length), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    targets = torch.full((len(items), length, len(TYPES)), -100, dtype=torch.long)
    for j, item in enumerate(items):
        n = len(item['input_ids'])
        ids[j, :n] = torch.tensor(item['input_ids'])
        mask[j, :n] = 1
        targets[j, :n] = torch.from_numpy(item['targets'])
    return ids, mask, targets


def extend_bert_positions(encoder, max_length):
    """Preserve pretrained positions; initialize extra BERT positions by tiling.

    This enables longer inputs but does not itself teach long-context behavior.
    """
    old_length = encoder.config.max_position_embeddings
    if max_length <= old_length:
        return old_length
    if encoder.config.model_type != 'bert':
        raise ValueError('Position extension is implemented only for BERT encoders')
    embeddings = encoder.embeddings
    if getattr(embeddings, 'position_embedding_type', 'absolute') != 'absolute':
        raise ValueError('Only absolute BERT position embeddings can be extended')
    old = embeddings.position_embeddings
    new = nn.Embedding(max_length, old.embedding_dim, device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        indices = torch.arange(max_length, device=old.weight.device) % old_length
        new.weight.copy_(old.weight[indices])
    new.weight.requires_grad_(old.weight.requires_grad)
    embeddings.position_embeddings = new
    embeddings.register_buffer('position_ids', torch.arange(max_length, device=old.weight.device).expand((1, -1)), persistent=False)
    embeddings.register_buffer('token_type_ids', torch.zeros((1, max_length), device=old.weight.device, dtype=torch.long), persistent=False)
    encoder.config.max_position_embeddings = max_length
    return old_length


class MultiBIO(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(encoder.config.hidden_size, len(TYPES) * 3)

    def forward(self, ids, mask):
        states = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        return self.head(self.dropout(states)).reshape(*ids.shape, len(TYPES), 3)

    def save(self, path, tokenizer, config):
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(path / 'encoder')
        tokenizer.save_pretrained(path / 'encoder')
        torch.save(self.head.state_dict(), path / 'head.pt')
        (path / 'experiment.json').write_text(json.dumps(config, indent=2))


def collect_logits(model, rows, tokenizer, args):
    """Choose central-window logits per absolute token; never truncate the document."""
    model.eval()
    outputs = []
    with torch.inference_mode():
        for row in rows:
            windows = encode(row, tokenizer, args.max_length, args.stride)
            tokens = {}
            for begin in range(0, len(windows), args.batch_size):
                batch = windows[begin:begin + args.batch_size]
                ids, mask, _ = collate(batch, tokenizer.pad_token_id)
                logits = model(ids.to(args.device), mask.to(args.device)).float().cpu().numpy()
                for window, scores in zip(batch, logits):
                    n = len(window['input_ids'])
                    for i, (s, e) in enumerate(window['offsets']):
                        if s == e:
                            continue
                        centrality = min(i, n - 1 - i)
                        if (s, e) not in tokens or centrality > tokens[s, e][0]:
                            tokens[s, e] = (centrality, scores[i])
            offsets = sorted(tokens)
            outputs.append((offsets, np.stack([tokens[o][1] for o in offsets]) if offsets else np.empty((0,len(TYPES),3))))
    return outputs


def decode(offsets, logits, bias=0.0):
    scores = logits.copy()
    scores[:, :, 1:] += bias
    tags = scores.argmax(-1)
    result = []
    for c, label in enumerate(TYPES):
        start = end = None
        for (s, e), tag in zip(offsets, tags[:, c]):
            if tag == 0 or tag == 1:
                if start is not None:
                    result.append({'start': start, 'end': end, 'label': label})
                    start = end = None
            if tag:
                if start is None:
                    start = s  # Recover orphan I, including window boundaries.
                end = e
        if start is not None:
            result.append({'start': start, 'end': end, 'label': label})
    return result


def report(rows, outputs, bias, target):
    metrics = span_metrics(rows, [decode(*x, bias) for x in outputs])
    required = {label: metrics['by_label'].get(label, {}) for label in sorted(TASK_REQUIRED_LABELS)}
    metrics['required_labels_below_target'] = [k for k,v in required.items() if v.get('recall',0) < target or v.get('tp',0)+v.get('fn',0)==0]
    metrics['required_labels_all_pass'] = not metrics['required_labels_below_target']
    metrics['target_recall'] = target
    return metrics


def selection_key(metrics, precision_floor):
    m = metrics['exact_span_micro']
    return (m['precision'] >= precision_floor, m['recall'] >= metrics['target_recall'],
            m['precision'] if m['recall'] >= metrics['target_recall'] else m['recall'], m['f1'])


def check_splits(splits):
    seen = {}
    for name, rows in splits.items():
        for row in rows:
            key = hashlib.sha256(' '.join(row['text'].casefold().split()).encode()).hexdigest()
            if key in seen and seen[key] != name:
                raise ValueError(f'Text leakage between {seen[key]} and {name}')
            seen[key] = name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='cointegrated/rubert-tiny2')
    p.add_argument('--corpus', type=Path, default=ROOT/'ru_gliner_hybrid_v4/data/hybrid')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-length', type=int, default=384)
    p.add_argument('--stride', type=int, default=96)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--extend-positions', action='store_true')
    p.add_argument('--gradient-checkpointing', action='store_true')
    p.add_argument('--gradient-accumulation-steps', type=int, default=1)
    p.add_argument('--device', default='cuda')
    p.add_argument('--target-recall', type=float, default=.95)
    p.add_argument('--precision-floor', type=float, default=.90)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        p.error('epochs, batch-size and gradient-accumulation-steps must be positive')
    if not 0 <= args.stride < args.max_length - 2:
        p.error('stride must be smaller than the token window')
    args.out.mkdir(parents=True, exist_ok=True)
    splits = {s: read_jsonl(args.corpus/f'{s}.jsonl') for s in ('train','dev','test')}
    check_splits(splits)
    if args.smoke:
        splits = {s: rows[:8] for s,rows in splits.items()}
        args.epochs = 1
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    encoder = AutoModel.from_pretrained(args.model)
    native_max_length = encoder.config.max_position_embeddings
    if args.max_length > native_max_length:
        if not args.extend_positions:
            p.error(f'max-length exceeds native limit {native_max_length}; use --extend-positions for BERT')
        extend_bert_positions(encoder, args.max_length)
    tokenizer.model_max_length = encoder.config.max_position_embeddings
    if args.gradient_checkpointing:
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        encoder.config.use_cache = False
    model = MultiBIO(encoder).to(args.device)
    train = [w for row in splits['train'] for w in encode(row, tokenizer, args.max_length, args.stride)]
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda x: collate(x, tokenizer.pad_token_id))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    updates_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(.1*updates_per_epoch*args.epochs), updates_per_epoch*args.epochs)
    config = {k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(native_max_position_embeddings=native_max_length,
                  position_initialization='tiled' if args.max_length > native_max_length else 'pretrained',
                  labels=TYPES, corpus_sha256={s:sha256_file(args.corpus/f'{s}.jsonl') for s in splits},
                  encoder_revision=getattr(model.encoder.config,'_commit_hash',None),
                  torch_version=torch.__version__)
    scaler = torch.amp.GradScaler('cuda', enabled=args.fp16 and args.device.startswith('cuda'))
    best = None
    started = time.time()
    for epoch in range(args.epochs):
        model.train(); losses=[]
        optimizer.zero_grad(set_to_none=True)
        for step,(ids,mask,targets) in enumerate(loader):
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=scaler.is_enabled()):
                logits = model(ids.to(args.device), mask.to(args.device))
                loss = nn.functional.cross_entropy(logits.reshape(-1,3),targets.to(args.device).reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            group_start = (step // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
            group_size = min(args.gradient_accumulation_steps, len(loader) - group_start)
            scaler.scale(loss / group_size).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())
            if step % 100 == 0:
                print(json.dumps({'epoch':epoch+1,'step':step,'loss':loss.item()}),flush=True)
        outputs = collect_logits(model, splits['dev'], tokenizer, args)
        for bias in (0.0, 0.5, 1.0, 1.5, 2.0):
            metrics = report(splits['dev'], outputs, bias, args.target_recall)
            key = selection_key(metrics,args.precision_floor)
            if best is None or key > best:
                best = key
                config.update(epoch=epoch+1,bias=bias,selection_split='dev')
                model.save(args.out,tokenizer,config)
                (args.out/'dev_metrics.json').write_text(json.dumps(metrics,indent=2))
        summary = {'epoch':epoch+1,'loss':sum(losses)/len(losses),'elapsed_s':time.time()-started,
                   'selected_epoch':config['epoch'],'selected_bias':config['bias']}
        with (args.out/'training_log.jsonl').open('a') as log:
            log.write(json.dumps(summary)+'\n')
        print(json.dumps(summary),flush=True)
    # Test is evaluated once after checkpoint and decoding bias selection on dev.
    del model, encoder, optimizer, scheduler, logits, loss
    if args.device.startswith('cuda'): torch.cuda.empty_cache()
    model = MultiBIO(AutoModel.from_pretrained(args.out/'encoder',local_files_only=True))
    model.head.load_state_dict(torch.load(args.out/'head.pt',map_location='cpu',weights_only=True))
    model.to(args.device)
    metrics = report(splits['test'],collect_logits(model,splits['test'],tokenizer,args),config['bias'],args.target_recall)
    metrics['smoke_only'] = args.smoke
    (args.out/'test_metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps({'test':metrics['exact_span_micro'],'required_labels_all_pass':metrics['required_labels_all_pass']}),flush=True)

if __name__ == '__main__':
    main()
