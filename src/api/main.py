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

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ..core.config import settings
from ..core.engine import InferenceEngine
from .correlation_service import CorrelationService
from .metrics import MetricsService
from .schemas import ProcessRequest, ProcessResponse

logger = logging.getLogger("pii.api")

# --- shared singletons ---
_engine: InferenceEngine | None = None
_metrics = MetricsService()


def get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine


def get_store() -> Any:
    """Return the correlation store shared with the inference engine.

    The engine owns the store (it writes correlations); the API reads from the
    same instance so masking/unmasking correlate correctly even with the
    in-memory backend.
    """
    return get_engine().store


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


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Return a clean JSON 500 for any unhandled error (no stack leak)."""
    logger.exception("unhandled error: %s", exc)
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": "internal error"})


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
    return _metrics.snapshot()


@app.post("/process", response_model=ProcessResponse)
async def process(req: ProcessRequest, request: Request, response: Response) -> ProcessResponse:
    start = time.monotonic()
    correlation = CorrelationService(get_store())

    # Correlation checks. The memory store is a fast dict lookup; the redis
    # store is wrapped in an executor to avoid blocking the event loop.
    record = await correlation.get(req.payload_id)
    if record is not None:
        # Unmasking: the payload is the mask we produced earlier.
        if record.get("masked") == req.payload:
            _metrics.record(unmasking=True, latency_s=time.monotonic() - start)
            return ProcessResponse(result=record["original"])
        # Masking retry with the same id -> return the stored mask.
        _metrics.record(masking=True, latency_s=time.monotonic() - start)
        return ProcessResponse(result=record["masked"])

    # New masking request. Idempotent retry while the job is in flight.
    cached = await correlation.get_result(req.payload_id)
    if cached is not None:
        _metrics.record(masking=True, latency_s=time.monotonic() - start)
        return ProcessResponse(result=cached)

    # Submit to the inference engine and await the result.
    try:
        result = await get_engine().submit(req.payload_id, req.payload, timeout_s=settings.request_timeout_s)
    except Exception:  # noqa: BLE001 - surface as a 500, never leak internals
        _metrics.record_error()
        logger.exception("inference engine failed for payload_id=%s", req.payload_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="inference failed")
    if result is None:
        _metrics.record_overloaded()
        response.headers["Retry-After"] = "1"
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="overloaded, retry later")

    _metrics.record(masking=True, latency_s=time.monotonic() - start)
    return ProcessResponse(result=result)