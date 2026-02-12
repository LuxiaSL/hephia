"""
core/routes/chat.py

Chat endpoints — OpenAI-compat and simple alias.
"""

from typing import Optional
from fastapi import APIRouter, Request, HTTPException, Body, Query

from config import Config
from shared_models.api_models import (
    ChatRequest,
    ChatHistoryMessage,
    ChatHistoryResponse,
    SimpleChatRequest,
    SimpleChatResponse,
)
from loggers import SystemLogger

router = APIRouter()


@router.post("/v1/chat/completions")
async def handle_conversation(request: Request, body: ChatRequest = Body(...)):
    """Chat endpoint — routes user messages through the Mind layer (OpenAI compat)."""
    mind = request.app.state.mind
    try:
        user_message = ""
        for msg in reversed(body.messages):
            if msg.role == "user":
                user_message = msg.content
                break

        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        response = await mind.process_message(user_message)

        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response.content,
                },
                "finish_reason": "stop",
            }],
            "model": Config.get_pet_model(),
            "usage": {},
        }
    except HTTPException:
        raise
    except Exception as e:
        SystemLogger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/history", response_model=ChatHistoryResponse)
async def get_chat_history(request: Request, limit: Optional[int] = Query(None)):
    """Return persisted conversation history."""
    mind = request.app.state.mind
    try:
        history = mind.conversation.get_history(limit=limit)
        messages = [
            ChatHistoryMessage(role=msg.role, content=msg.content)
            for msg in history
        ]
        return ChatHistoryResponse(messages=messages)
    except Exception as e:
        SystemLogger.error(f"Chat history endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def simple_chat(request: Request, body: SimpleChatRequest = Body(...)):
    """Simple chat endpoint — accepts a message string, returns pet response."""
    mind = request.app.state.mind
    try:
        response = await mind.process_message(body.message)
        return SimpleChatResponse(
            content=response.content,
            memories_used=response.memories_used,
            was_task=response.was_task,
        )
    except Exception as e:
        SystemLogger.error(f"Simple chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
