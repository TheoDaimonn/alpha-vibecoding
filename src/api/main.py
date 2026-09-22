"""FastAPI application implementing the POST /process contract.

Flow for a request:
  1. If the payload_id is already known and the payload equals the stored mask,
     this is the unmasking step -> return the stored original.
  2. If the payload_id is already known (masking retry) -> return the stored mask.
  3. Otherwise this is a new masking request -> submit to the inference engine
     (in-process by default) and return the mask.

The endpoint is fully async so it can hold thousands of concurrent connections.
It is idempotent by payload_id and returns 429 when the system is overloaded.
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ..core.config import settings
from ..core.engine import InferenceEngine
from .schemas import ProcessRequest, ProcessResponse

logger = logging.getLogger("pii.api")

# --- shared singletons ---
_engine: InferenceEngine | None = None

# --- metrics ---
_metrics: dict[str, Any] = {
    "total": 0,
    "masking": 0,
    "unmasking": 0,
    "errors": 0,
    "overloaded": 0,
    "latency_sum_s": 0.0,
}


def get_store() -> Any:
    """Return the correlation store shared with the inference engine.

    The engine owns the store (it writes correlations); the API reads from the
    same instance so masking/unmasking correlate correctly even with the
    in-memory backend.
    """
    return get_engine().store


def get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine


def _authorized(request: Request) -> bool:
    if not settings.api_keys:
        return True
    key = request.headers.get("X-API-Key", "")
    return key in settings.api_keys


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await get_engine().start()
    yield
    await get_engine().stop()


app = FastAPI(title="PII Security Module", version="1.0.0", lifespan=lifespan)

# Manual test stand (static UI).
app.mount("/ui", StaticFiles(directory="src/api/static", html=True), name="ui")


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.url.path == "/process" and not _authorized(request):
        return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": "unauthorized"})
    return await call_next(request)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    m = dict(_metrics)
    if m["total"]:
        m["avg_latency_s"] = round(m["latency_sum_s"] / m["total"], 4)
    return m


@app.post("/process", response_model=ProcessResponse)
async def process(req: ProcessRequest, request: Request, response: Response) -> ProcessResponse:
    start = time.monotonic()
    store = get_store()
    _metrics["total"] += 1

    # Correlation checks. The memory store is a fast dict lookup; the redis
    # store is wrapped in an executor to avoid blocking the event loop.
    if settings.correlation_store == "redis":
        loop = asyncio.get_running_loop()
        correlation = await loop.run_in_executor(None, store.get, req.payload_id)
    else:
        correlation = store.get(req.payload_id)
    if correlation is not None:
        # Unmasking: the payload is the mask we produced earlier.
        if correlation.get("masked") == req.payload:
            _metrics["unmasking"] += 1
            _metrics["latency_sum_s"] += time.monotonic() - start
            return ProcessResponse(result=correlation["original"])
        # Masking retry with the same id -> return the stored mask.
        _metrics["masking"] += 1
        _metrics["latency_sum_s"] += time.monotonic() - start
        return ProcessResponse(result=correlation["masked"])

    # New masking request. Idempotent retry while the job is in flight.
    if settings.correlation_store == "redis":
        cached = await loop.run_in_executor(None, store.get_result, req.payload_id)
    else:
        cached = store.get_result(req.payload_id)
    if cached is not None:
        _metrics["masking"] += 1
        _metrics["latency_sum_s"] += time.monotonic() - start
        return ProcessResponse(result=cached)

    # Submit to the inference engine and await the result.
    result = await get_engine().submit(req.payload_id, req.payload, timeout_s=settings.request_timeout_s)
    if result is None:
        _metrics["overloaded"] += 1
        response.headers["Retry-After"] = "1"
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="overloaded, retry later")

    _metrics["masking"] += 1
    _metrics["latency_sum_s"] += time.monotonic() - start
    return ProcessResponse(result=result)