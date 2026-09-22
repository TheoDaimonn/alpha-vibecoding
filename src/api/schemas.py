"""Pydantic request/response models for the /process contract."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ProcessRequest(BaseModel):
    payload: str = Field(..., description="Text to process (original on masking, mask on unmasking)")
    payload_id: str = Field(..., description="Correlation id shared by a masking/unmasking pair")


class ProcessResponse(BaseModel):
    result: str = Field(..., description="Masked text on masking, original text on unmasking")