"""Entrypoint for the HTTP API (uvicorn)."""
from __future__ import annotations

import logging
import os

import uvicorn

from .core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def main() -> None:
    workers = int(os.getenv("API_WORKERS", "1"))
    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        workers=workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()