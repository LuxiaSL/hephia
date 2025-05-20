# hephia_project/shared_models/tui_events.py
from pydantic import BaseModel, field_validator
from typing import List, Dict, Any, Optional

# For raw_state (Cognitive Processing Panel)
class TUIMessage(BaseModel):
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

# For context (System State Panel)
class TUIMood(BaseModel):
    name: Optional[str] = None
    valence: float
    arousal: float

class TUINeed(BaseModel):
    satisfaction: float

class TUIBehavior(BaseModel):
    name: Optional[str] = None
    active: bool

class TUIEmotionalStateItem(BaseModel):
    name: Optional[str] = None
    intensity: float
    valence: float
    arousal: float

class TUISystemContext(BaseModel):
    mood: Optional[TUIMood] = None
    needs: Optional[Dict[str, TUINeed]] = None
    behavior: Optional[TUIBehavior] = None
    emotional_state: Optional[List[TUIEmotionalStateItem]] = None

# Main payload for TUI WebSocket messages
class TUIDataPayload(BaseModel):
    """
    Represents the complete data snapshot to be sent to the TUI for a screen refresh.
    """
    recent_messages: Optional[List[TUIMessage]] = None
    system_context: Optional[TUISystemContext] = None
    cognitive_summary: Optional[str] = None
    current_model_name: Optional[str] = "N/A"
    
    @field_validator('recent_messages', mode='before')
    @classmethod
    def ensure_recent_messages_is_list(cls, v):
        if v is None:
            return []
        return v

    @field_validator('cognitive_summary', mode='before')
    @classmethod
    def ensure_cognitive_summary_is_str(cls, v):
        if v is None:
            return ""
        return v


class TUIWebSocketMessage(BaseModel):
    """
    The wrapper for all messages sent to the TUI over WebSockets.
    """
    event_type: str
    payload: TUIDataPayload
    timestamp: str