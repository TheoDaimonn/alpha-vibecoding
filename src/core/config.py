"""Service configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _str(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    # --- HTTP API ---
    host: str = field(default_factory=lambda: _str("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _int("PORT", 8000))
    request_timeout_s: float = field(default_factory=lambda: _float("REQUEST_TIMEOUT_S", 9.0))
    poll_interval_s: float = field(default_factory=lambda: _float("POLL_INTERVAL_S", 0.002))

    # --- Redis ---
    redis_url: str = field(default_factory=lambda: _str("REDIS_URL", "redis://localhost:6379/0"))
    result_ttl_s: int = field(default_factory=lambda: _int("RESULT_TTL_S", 3600))

    # Correlation store: memory (fast, default) | redis (shared)
    correlation_store: str = field(default_factory=lambda: _str("CORRELATION_STORE", "memory"))

    # --- Model / worker ---
    detector: str = field(default_factory=lambda: _str("DETECTOR", "student"))
    student_model_path: str = field(
        default_factory=lambda: _str("STUDENT_MODEL_PATH", "artifacts/student-pii.pt")
    )
    device: str = field(default_factory=lambda: _str("DEVICE", "cpu"))
    model_batch_size: int = field(default_factory=lambda: _int("MODEL_BATCH_SIZE", 16))
    worker_batch_size: int = field(default_factory=lambda: _int("WORKER_BATCH_SIZE", 32))
    worker_batch_timeout_s: float = field(default_factory=lambda: _float("WORKER_BATCH_TIMEOUT_S", 0.02))
    worker_threads: int = field(default_factory=lambda: _int("WORKER_THREADS", 4))
    threshold: float = field(default_factory=lambda: _float("THRESHOLD", 0.5))
    # Post-process: filter public figures / public addresses (not PII).
    postprocess: bool = field(default_factory=lambda: _str("POSTPROCESS", "0") == "1")

    # --- Auth (allowlist of consumer systems) ---
    api_keys: tuple[str, ...] = field(default_factory=lambda: tuple(k for k in _str("API_KEYS", "").split(",") if k))


settings = Settings()