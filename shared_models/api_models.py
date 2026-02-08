"""
shared_models/api_models.py

Pydantic request/response models for all REST API routes.
"""

from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field


# --- Chat ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: bool = False

class SimpleChatRequest(BaseModel):
    message: str

class SimpleChatResponse(BaseModel):
    content: str
    memories_used: List[str] = Field(default_factory=list)
    was_task: bool = False


# --- State / Actions ---

class ActionRequest(BaseModel):
    action: str = Field(..., description="Name of the action to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    message: Optional[str] = None

class ActionResponse(BaseModel):
    success: bool
    message: str
    state_changes: Dict[str, Any]

class ActionStatus(BaseModel):
    last_execution: float
    total_executions: int
    successful_executions: int
    failed_executions: int
    on_cooldown: bool
    remaining_cooldown: float

class ActionInfo(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    status: ActionStatus


# --- Memory ---

class MemoryResponse(BaseModel):
    id: str
    content: str
    timestamp: float
    strength: float
    relevance: float = 0.0


# --- Notes ---

class NoteCreateRequest(BaseModel):
    content: str
    tags: List[str] = Field(default_factory=list)
    sticky: bool = False

class NoteUpdateRequest(BaseModel):
    content: Optional[str] = None
    tags: Optional[List[str]] = None
    sticky: Optional[bool] = None

class NoteResponse(BaseModel):
    id: str
    content: str
    created_at: str
    updated_at: str
    context: Optional[str] = None
    sticky: bool = False
    tags: List[str] = Field(default_factory=list)


# --- Worker ---

class WorkerTaskRequest(BaseModel):
    task: str
    context: Optional[str] = None

class WorkerTaskResponse(BaseModel):
    task_id: str

class WorkerTaskStatus(BaseModel):
    task_id: str
    status: str  # "pending", "running", "completed", "failed"
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: float
    completed_at: Optional[float] = None


# --- Settings ---

class PetSettings(BaseModel):
    pet_name: str = "Hephia"
    pet_model: str = "haiku-4.5"
    worker_model: str = "opus-4.6"
    personality_prompt: str = ""
    memory_significance_threshold: float = 0.625
    max_conversation_turns: int = 50

class SettingsResponse(BaseModel):
    settings: PetSettings
    available_models: List[str] = Field(default_factory=list)
    version: str = ""
