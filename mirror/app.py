"""Сервис-зеркало контракта /process (ТЗ хакатона).

Принимает запросы {payload, payload_id}, залогирует каждое тело в JSONL-файл
и возвращает payload как есть. Повторный запрос с тем же payload_id возвращает
сохранённый исходный payload (эмуляция идемпотентного демаскирования).

Все обработчики асинхронные: лог пишется через await в отдельном потоке
(asyncio.to_thread), чтобы дисковый I/O не блокировал event loop.
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.environ.get("LOG_PATH", os.path.join(APP_DIR, "logs", "requests.log"))

_log_lock = asyncio.Lock()
_store_lock = asyncio.Lock()
_store: dict[str, str] = {}

app = FastAPI(title="PII mirror stub")


class ProcessRequest(BaseModel):
    payload: str
    payload_id: str


class ProcessResponse(BaseModel):
    result: str


def _append_log_line_sync(line: str) -> None:
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


async def _log_request(payload_id: str, payload: str) -> None:
    line = json.dumps(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "payload_id": payload_id,
            "payload": payload,
        },
        ensure_ascii=False,
    )
    async with _log_lock:
        await asyncio.to_thread(_append_log_line_sync, line)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/process", response_model=ProcessResponse)
async def process(req: ProcessRequest) -> ProcessResponse:
    await _log_request(req.payload_id, req.payload)
    async with _store_lock:
        original = _store.get(req.payload_id)
        if original is None:
            _store[req.payload_id] = req.payload
            original = req.payload
    return ProcessResponse(result=original)
