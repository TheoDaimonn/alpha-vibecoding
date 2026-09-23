"""Service configuration loaded from environment variables."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _optional_int(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    return int(value) if value else None


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
    rubert_model_path: str = field(
        default_factory=lambda: _str("RUBERT_MODEL_PATH", "artifacts/rubert-tiny2-fine-tuning")
    )
    distil_model_path: str = field(
        default_factory=lambda: _str("DISTIL_MODEL_PATH", "artifacts/rubert-distil")
    )
    # Empty window settings use the checkpoint configuration.
    rubert_max_len: int | None = field(default_factory=lambda: _optional_int("RUBERT_MAX_LEN"))
    rubert_stride: int | None = field(default_factory=lambda: _optional_int("RUBERT_STRIDE"))
    onnx_intra_threads: int = field(default_factory=lambda: _int("ONNX_INTRA_THREADS", 1))
    onnx_inter_threads: int = field(default_factory=lambda: _int("ONNX_INTER_THREADS", 1))
    device: str = field(default_factory=lambda: _str("DEVICE", "cpu"))
    model_batch_size: int = field(default_factory=lambda: _int("MODEL_BATCH_SIZE", 16))
    worker_batch_size: int = field(default_factory=lambda: _int("WORKER_BATCH_SIZE", 32))
    worker_batch_timeout_s: float = field(default_factory=lambda: _float("WORKER_BATCH_TIMEOUT_S", 0.02))
    worker_threads: int = field(default_factory=lambda: _int("WORKER_THREADS", 4))
    threshold: float = field(default_factory=lambda: _float("THRESHOLD", 0.5))
    # Post-process: filter public figures / public addresses (not PII).
    postprocess: bool = field(default_factory=lambda: _str("POSTPROCESS", "0") == "1")

    worker_queue_size: int = field(default_factory=lambda: _int("WORKER_QUEUE_SIZE", 256))
    worker_queue_chars: int = field(default_factory=lambda: _int("WORKER_QUEUE_CHARS", 4_000_000))

    # --- Auth (allowlist of consumer systems) ---
    api_keys: tuple[str, ...] = field(default_factory=lambda: tuple(k.strip() for k in _str("API_KEYS", "").split(",") if k.strip()))

    def __post_init__(self) -> None:
        for name in ("port", "result_ttl_s", "model_batch_size", "worker_batch_size",
                     "worker_threads", "worker_queue_size", "worker_queue_chars",
                     "onnx_intra_threads", "onnx_inter_threads"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("request_timeout_s", "poll_interval_s", "worker_batch_timeout_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name != "worker_batch_timeout_s" and value == 0):
                raise ValueError(f"{name} has an invalid duration")
        if self.port > 65535:
            raise ValueError("port must be <= 65535")
        if self.correlation_store not in {"memory", "redis"}:
            raise ValueError("unknown correlation store backend")


settings = Settings()