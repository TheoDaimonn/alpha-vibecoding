"""FastAPI application implementing the POST /process contract.

Flow for a request:
  1. A saved mask with its payload_id returns the original.
  2. Other requests for that id return the saved mask (idempotent retries).
  3. Otherwise this is a new masking request -> submit to the inference engine
     (in-process by default) and return the mask.

The endpoint is fully async so it can hold thousands of concurrent connections.
It is idempotent by payload_id and returns 429 when the system is overloaded.
"""
from __future__ import annotations

import asyncio
import logging
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
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
    return any(secrets.compare_digest(key.encode(), allowed.encode()) for allowed in settings.api_keys)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await get_engine().start()
    try:
        yield
    finally:
        await get_engine().stop()


app = FastAPI(title="PII Security Module", version="1.0.0", lifespan=lifespan)

# Manual test stand (static UI).
app.mount("/ui", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Return a clean JSON 500 for any unhandled error (no stack leak)."""
    logger.error("unhandled API error (%s)", type(exc).__name__)
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": "internal error"})


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.url.path == "/process" and not _authorized(request):
        return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": "unauthorized"})
    return await call_next(request)


@app.get("/health")
async def health() -> dict[str, Any]:
    engine = get_engine()
    if not engine.running:
        raise HTTPException(status_code=503, detail="engine unavailable")
    try:
        ready = await asyncio.wait_for(CorrelationService(engine.store).ping(), timeout=2.5)
    except Exception:  # noqa: BLE001 - readiness returns no storage credentials
        ready = False
    if not ready:
        raise HTTPException(status_code=503, detail="storage unavailable")
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return _metrics.snapshot()


@app.post("/process", response_model=ProcessResponse)
async def process(req: ProcessRequest, request: Request, response: Response) -> ProcessResponse:
    start = time.monotonic()
    payload_id = req.payload_id
    correlation = CorrelationService(get_store())
    try:
        record = await correlation.get(payload_id)
        if record is not None:
            if req.payload == record["masked"]:
                _metrics.record(unmasking=True, latency_s=time.monotonic() - start)
                return ProcessResponse(result=record["original"])
            _metrics.record(masking=True, latency_s=time.monotonic() - start)
            return ProcessResponse(result=record["masked"])
        result = await get_engine().submit(payload_id, req.payload, timeout_s=settings.request_timeout_s)
    except HTTPException:
        _metrics.record_error()
        raise
    except Exception as exc:  # noqa: BLE001 - API boundary must not expose internal exceptions
        _metrics.record_error()
        # Exception messages from models/storage may contain user input or credentials.
        logger.error("processing failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=500, detail="processing failed") from None
    if result is None:
        _metrics.record_overloaded()
        raise HTTPException(
            status_code=429,
            detail="overloaded, retry later",
            headers={"Retry-After": "1"},
        )
    _metrics.record(masking=True, latency_s=time.monotonic() - start)
    return ProcessResponse(result=result)
