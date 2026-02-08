"""
core/routes/notes.py

Notes CRUD + FTS search REST endpoints.
"""

from typing import Dict, List, Optional

from fastapi import APIRouter, Request, HTTPException, Query

from shared_models.api_models import (
    NoteCreateRequest,
    NoteUpdateRequest,
    NoteResponse,
)
from loggers import SystemLogger

router = APIRouter(prefix="/notes", tags=["notes"])


@router.post("", response_model=NoteResponse, status_code=201)
async def create_note(request: Request, body: NoteCreateRequest):
    """Create a new note with optional tags and sticky flag."""
    notes = request.app.state.notes
    state_bridge = request.app.state.state_bridge
    try:
        # Capture current state context for the note
        context_data = await state_bridge.get_api_context(use_memory_emotions=False)
        mood = context_data.get("mood", {})
        context_str = None
        if mood:
            mood_name = mood.get("name", "neutral")
            context_str = f"mood:{mood_name}"

        return notes.create(
            content=body.content,
            tags=body.tags,
            sticky=body.sticky,
            context=context_str,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        SystemLogger.error(f"Note create error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search", response_model=List[NoteResponse])
async def search_notes(
    request: Request,
    q: str = Query(..., min_length=1),
    limit: int = Query(default=10, ge=1, le=100),
):
    """Full-text search across notes."""
    notes = request.app.state.notes
    try:
        return notes.search(query=q, limit=limit)
    except Exception as e:
        SystemLogger.error(f"Note search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tags")
async def list_tags(request: Request) -> Dict[str, int]:
    """List all tags with their usage counts."""
    notes = request.app.state.notes
    try:
        return notes.list_tags()
    except Exception as e:
        SystemLogger.error(f"Note tags error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=List[NoteResponse])
async def list_notes(
    request: Request,
    tag: Optional[str] = Query(default=None),
    sticky_only: bool = Query(default=False),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """List notes with optional filtering by tag or sticky status."""
    notes = request.app.state.notes
    try:
        return notes.list_notes(
            tag=tag,
            sticky_only=sticky_only,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        SystemLogger.error(f"Note list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{note_id}", response_model=NoteResponse)
async def get_note(note_id: str, request: Request):
    """Get a single note by ID."""
    notes = request.app.state.notes
    try:
        result = notes.get(note_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Note not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        SystemLogger.error(f"Note get error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{note_id}", response_model=NoteResponse)
async def update_note(note_id: str, request: Request, body: NoteUpdateRequest):
    """Update an existing note."""
    notes = request.app.state.notes
    try:
        result = notes.update(
            note_id=note_id,
            content=body.content,
            tags=body.tags,
            sticky=body.sticky,
        )
        if result is None:
            raise HTTPException(status_code=404, detail="Note not found")
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        SystemLogger.error(f"Note update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{note_id}")
async def delete_note(note_id: str, request: Request):
    """Delete a note."""
    notes = request.app.state.notes
    try:
        deleted = notes.delete(note_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Note not found")
        return {"success": True, "message": "Note deleted"}
    except HTTPException:
        raise
    except Exception as e:
        SystemLogger.error(f"Note delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
