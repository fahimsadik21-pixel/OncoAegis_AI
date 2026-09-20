"""Authenticated ChatGPT-style conversation history endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.schemas.chat_history import ChatDeleteResponse, ChatTurnRequest
from app.security.dependencies import get_current_user
from app.storage.chat_history import (
    ChatConversationNotFound,
    ChatHistoryError,
    ChatHistoryStore,
)


router = APIRouter(prefix="/chat", tags=["chat-history"])
_store = ChatHistoryStore()


@router.get("/history")
def list_chat_history(
    limit: int = Query(50, ge=1, le=100),
    current_user=Depends(get_current_user),
) -> list[dict[str, object]]:
    return _store.list_conversations(user_id=current_user.user_id, limit=limit)


@router.get("/history/{conversation_id}")
def get_chat_history(
    conversation_id: str,
    current_user=Depends(get_current_user),
) -> dict[str, object]:
    try:
        return _store.get_conversation(
            user_id=current_user.user_id,
            conversation_id=conversation_id,
        )
    except ChatConversationNotFound as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc


@router.post("/history")
def append_chat_turn(
    request: ChatTurnRequest,
    current_user=Depends(get_current_user),
) -> dict[str, object]:
    try:
        return _store.append_turn(
            user_id=current_user.user_id,
            conversation_id=request.conversation_id,
            title=request.title,
            user_message=request.user_message,
            assistant_message=request.assistant_message,
            kind=request.kind,
            metadata=request.metadata,
        )
    except ChatHistoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/history/{conversation_id}", response_model=ChatDeleteResponse)
def delete_chat_history(
    conversation_id: str,
    current_user=Depends(get_current_user),
) -> ChatDeleteResponse:
    try:
        _store.delete_conversation(
            user_id=current_user.user_id,
            conversation_id=conversation_id,
        )
    except ChatConversationNotFound as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc
    return ChatDeleteResponse()
