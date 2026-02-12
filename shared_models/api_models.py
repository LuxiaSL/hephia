"""
shared_models/api_models.py

Pydantic request/response models for all REST API routes.
"""

from typing import List, Dict, Any, Optional, Literal

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

class AgentProgressEntry(BaseModel):
    type: Literal["text", "tool_use", "tool_result", "thinking", "error"] = "text"
    content: str = ""
    timestamp: float = 0.0
    tool_name: Optional[str] = None
    tool_input_summary: Optional[str] = None

class AgentQuestionOption(BaseModel):
    label: str
    description: str

class AgentQuestion(BaseModel):
    question: str
    options: List[AgentQuestionOption] = Field(default_factory=list)
    multi_select: bool = False

class WorkerTaskStatus(BaseModel):
    task_id: str
    status: str  # "pending", "running", "completed", "failed", "awaiting_input"
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: float
    completed_at: Optional[float] = None
    progress: List[AgentProgressEntry] = Field(default_factory=list)
    cost_usd: Optional[float] = None
    num_turns: Optional[int] = None
    pending_questions: List[AgentQuestion] = Field(default_factory=list)

class WorkerReplyRequest(BaseModel):
    message: str

class WorkerClearRequest(BaseModel):
    completed_only: bool = True

class WorkerStarResponse(BaseModel):
    node_id: Optional[str] = None
    error: Optional[str] = None


# --- Chat History ---

class ChatHistoryMessage(BaseModel):
    role: str
    content: str

class ChatHistoryResponse(BaseModel):
    messages: List[ChatHistoryMessage]


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
