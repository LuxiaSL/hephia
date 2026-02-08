"""
core/routes/settings.py

Runtime settings endpoints.
"""

from fastapi import APIRouter, Request, HTTPException, Body

from shared_models.api_models import PetSettings, SettingsResponse
from loggers import SystemLogger

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", response_model=SettingsResponse)
async def get_settings(request: Request):
    """Get current settings plus available models and version."""
    service = request.app.state.settings_service
    try:
        return service.get()
    except Exception as e:
        SystemLogger.error(f"Settings get error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("", response_model=SettingsResponse)
async def update_settings(request: Request, body: PetSettings = Body(...)):
    """Update runtime settings. Validates models against AVAILABLE_MODELS."""
    service = request.app.state.settings_service
    try:
        return service.update(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        SystemLogger.error(f"Settings update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
