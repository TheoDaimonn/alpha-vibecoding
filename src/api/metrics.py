"""Request metrics for the HTTP API.

Tracks per-process counters and latency. The counters are updated from the
async event loop (single-threaded), so a plain dict is safe; a lock is used
nonetheless to keep the class usable from worker threads if needed.
"""
from __future__ import annotations

import threading
import time
from typing import Any


class MetricsService:
    """Thread-safe counters and latency aggregation for /process requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total = 0
        self._masking = 0
        self._unmasking = 0
        self._errors = 0
        self._overloaded = 0
        self._latency_sum_s = 0.0

    def record(self, *, masking: bool = False, unmasking: bool = False, latency_s: float) -> None:
        with self._lock:
            self._total += 1
            self._latency_sum_s += latency_s
            if masking:
                self._masking += 1
            if unmasking:
                self._unmasking += 1

    def record_error(self) -> None:
        with self._lock:
            self._errors += 1

    def record_overloaded(self) -> None:
        with self._lock:
            self._overloaded += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            data: dict[str, Any] = {
                "total": self._total,
                "masking": self._masking,
                "unmasking": self._unmasking,
                "errors": self._errors,
                "overloaded": self._overloaded,
                "latency_sum_s": round(self._latency_sum_s, 4),
            }
            if self._total:
                data["avg_latency_s"] = round(self._latency_sum_s / self._total, 4)
            return data