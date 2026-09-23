"""Bounded fixed-rate /process benchmark. Synthetic data, unique IDs, no retries."""
import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import resource
import time
import uuid

import aiohttp


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered)-1, int((len(ordered)-1)*fraction))] if ordered else None


async def run(args):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, max(soft, 32768)), hard))
    total = int(args.rate * args.duration)
    prefix = 'bench-' + uuid.uuid4().hex
    connector = aiohttp.TCPConnector(limit=12000, limit_per_host=12000)
    statuses = Counter()
    latencies, successful, lags = [], [], []
    completed_in_window = 0
    pending = set()
    skipped = 0
    began = time.perf_counter()
    async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=args.timeout)) as session:
        async def request(i, scheduled):
            nonlocal completed_in_window
            started = time.perf_counter()
            lags.append(started - scheduled)
            text = f'Клиент Иванов Иван Иванович, email: user{i}@example.org, паспорт 4509 {i:06d}, телефон +7 900 123-45-67, ИНН 7707083893'
            ok = False
            try:
                async with session.post(args.url.rstrip('/')+'/process', json={'payload':text, 'payload_id':f'{prefix}-{i}'}) as response:
                    body = await response.read()
                    statuses[str(response.status)] += 1
                    if response.status == 200:
                        data = json.loads(body)
                        ok = isinstance(data.get('result'), str)
                        if not ok:
                            statuses['invalid_response'] += 1
            except Exception as exc:
                statuses[type(exc).__name__] += 1
            ended = time.perf_counter()
            latencies.append(ended-started)
            if ok:
                successful.append(ended-started)
                if ended-began <= args.duration:
                    completed_in_window += 1

        for i in range(total):
            scheduled = began + i / args.rate
            await asyncio.sleep(max(0, scheduled-time.perf_counter()))
            if len(pending) >= 12000:
                skipped += 1
                continue
            task = asyncio.create_task(request(i, scheduled))
            pending.add(task)
            task.add_done_callback(pending.discard)
        injection_s = time.perf_counter()-began
        if pending:
            await asyncio.gather(*pending)
    elapsed = time.perf_counter()-began
    return dict(url=args.url, rate=args.rate, duration_s=args.duration, scheduled=total,
        sent=len(latencies), client_skipped=skipped, statuses=dict(statuses), successful=len(successful),
        elapsed_s=elapsed, injection_s=injection_s, dispatch_rps=len(latencies)/injection_s,
        successful_rps_including_drain=len(successful)/elapsed,
        successful_rps_during_window=completed_in_window/args.duration,
        successful_within_1s=sum(x<=1 for x in successful),
        latency_s={f'p{int(p*100)}':percentile(successful,p) for p in (.5,.95,.99)},
        scheduling_lag_s={f'p{int(p*100)}':percentile(lags,p) for p in (.5,.95,.99)},
        max_latency_s=max(latencies,default=0), payload='Synthetic Russian PII, approximately 130 characters; unique IDs and emails; no retries')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--rate', type=int, default=1000)
    parser.add_argument('--duration', type=float, default=10)
    parser.add_argument('--timeout', type=float, default=12)
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args()
    if args.rate <= 0 or args.duration <= 0 or args.rate*args.duration>30000:
        parser.error('Use a positive rate/duration with at most 30000 requests per run')
    result=asyncio.run(run(args))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    main()
