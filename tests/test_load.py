"""Load test: fire N concurrent /process requests and measure latency.

Usage (service must be running, e.g. `docker compose up --build`):

    python -m tests.test_load
    # or
    pytest tests/test_load.py -s

Config via env vars:
    LOAD_URL          base URL, default http://localhost:8000
    LOAD_CONCURRENCY  number of concurrent requests, default 1000
    LOAD_COUNT        total requests, default 1000
    LOAD_TIMEOUT      per-request timeout seconds, default 10
    LOAD_RETRIES      retries per request on transient errors, default 2

Uses aiohttp with a shared connection pool for accurate high-concurrency
measurement. Sends `LOAD_CONCURRENCY` requests at once (one per payload_id) and
reports latency percentiles (p50/p90/p99/max) and achieved RPS.
"""
from __future__ import annotations

import asyncio
import os
import statistics
import time

import aiohttp

BASE_URL = os.getenv("LOAD_URL", "http://localhost:8000")
CONCURRENCY = int(os.getenv("LOAD_CONCURRENCY", "1000"))
COUNT = int(os.getenv("LOAD_COUNT", "1000"))
TIMEOUT = float(os.getenv("LOAD_TIMEOUT", "10"))
RETRIES = int(os.getenv("LOAD_RETRIES", "2"))


def _sample_text(i: int) -> str:
    return (
        f"Клиент Иванов Иван Иванович, email: user{i}@example.org, "
        f"паспорт 4509 {i:06d}, телефон +7 900 123-45-67, ИНН 7707083893"
    )


async def _one(session: aiohttp.ClientSession, i: int) -> float:
    payload_id = f"load-{i}"
    body = {"payload": _sample_text(i), "payload_id": payload_id}
    for attempt in range(RETRIES + 1):
        start = time.perf_counter()
        try:
            async with session.post(f"{BASE_URL}/process", json=body, timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as resp:
                elapsed = time.perf_counter() - start
                if resp.status == 200:
                    await resp.read()
                    return elapsed
                if resp.status == 429:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"request {i} failed: HTTP {resp.status}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if attempt >= RETRIES:
                raise RuntimeError(f"request {i} failed after retries: {exc}") from exc
            await asyncio.sleep(0.2 * (attempt + 1))
    raise RuntimeError(f"request {i} exhausted retries")


async def _run() -> list[float]:
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, limit_per_host=CONCURRENCY, force_close=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(CONCURRENCY)

        async def worker(i: int) -> float:
            async with sem:
                return await _one(session, i)

        tasks = [asyncio.create_task(worker(i)) for i in range(COUNT)]
        return await asyncio.gather(*tasks)


def main() -> None:
    print(f"Firing {COUNT} requests with concurrency {CONCURRENCY} -> {BASE_URL}/process")
    start = time.perf_counter()
    latencies = asyncio.run(_run())
    total = time.perf_counter() - start
    latencies.sort()
    n = len(latencies)
    rps = n / total

    def pct(p: float) -> float:
        return latencies[min(n - 1, int(p * n))]

    print("=" * 50)
    print(f"total requests : {n}")
    print(f"wall time      : {total:.3f}s")
    print(f"achieved RPS   : {rps:.1f}")
    print(f"min latency    : {latencies[0]*1000:.1f} ms")
    print(f"p50 latency    : {pct(0.50)*1000:.1f} ms")
    print(f"p90 latency    : {pct(0.90)*1000:.1f} ms")
    print(f"p99 latency    : {pct(0.99)*1000:.1f} ms")
    print(f"max latency    : {latencies[-1]*1000:.1f} ms")
    print(f"avg latency    : {statistics.mean(latencies)*1000:.1f} ms")
    print("=" * 50)


if __name__ == "__main__":
    main()