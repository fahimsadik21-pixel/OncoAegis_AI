"""Contracts for authenticated conversation history."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatTurnRequest(BaseModel):
    conversation_id: str | None = Field(default=None, max_length=80)
    title: str = Field(min_length=1, max_length=120)
    user_message: str = Field(min_length=1, max_length=30_000)
    assistant_message: str = Field(min_length=1, max_length=30_000)
    kind: Literal["text", "analysis"] = "text"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatDeleteResponse(BaseModel):
    status: Literal["deleted"] = "deleted"
