"""Entrypoint for the HTTP API (uvicorn)."""
from __future__ import annotations

import logging
import os

import uvicorn

from .core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def main() -> None:
    workers = int(os.getenv("API_WORKERS", "1"))
    if workers < 1:
        raise ValueError("API_WORKERS must be positive")
    if workers > 1 and settings.correlation_store == "memory":
        raise ValueError("multiple API workers require CORRELATION_STORE=redis")
    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        workers=workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()