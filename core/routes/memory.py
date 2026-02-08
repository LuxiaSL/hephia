"""
core/routes/memory.py

Memory browsing endpoints — delegates to PetCognitiveBridge.
"""

from typing import List

from fastapi import APIRouter, Request, HTTPException, Query

from shared_models.api_models import MemoryResponse
from loggers import SystemLogger

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("/recent", response_model=List[MemoryResponse])
async def get_recent_memories(
    request: Request,
    limit: int = Query(default=5, ge=1, le=50),
):
    """Get the most recent cognitive memories."""
    bridge = request.app.state.bridge
    try:
        memories = await bridge.get_recent_memories(limit=limit)
        return [
            MemoryResponse(
                id=m.id,
                content=m.content,
                timestamp=m.timestamp,
                strength=m.strength,
                relevance=m.relevance,
            )
            for m in memories
        ]
    except Exception as e:
        SystemLogger.error(f"Memory recent endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search", response_model=List[MemoryResponse])
async def search_memories(
    request: Request,
    q: str = Query(..., min_length=1),
    limit: int = Query(default=5, ge=1, le=50),
):
    """Search cognitive memories by query."""
    bridge = request.app.state.bridge
    try:
        memories = await bridge.retrieve_memories(query=q, limit=limit)
        return [
            MemoryResponse(
                id=m.id,
                content=m.content,
                timestamp=m.timestamp,
                strength=m.strength,
                relevance=m.relevance,
            )
            for m in memories
        ]
    except Exception as e:
        SystemLogger.error(f"Memory search endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
