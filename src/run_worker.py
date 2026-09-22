"""Entrypoint for an inference worker process.

Run one worker per model replica. Each worker loads its own model instance and
runs several consumer threads that batch jobs through the model.
"""
from __future__ import annotations

import logging

from .core.config import settings
from .core.worker import Worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def main() -> None:
    worker = Worker()
    worker.run()


if __name__ == "__main__":
    main()