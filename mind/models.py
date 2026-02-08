"""
mind/models.py

Pydantic data contracts shared across the mind layer.
"""

import time
from typing import List, Dict, Any, Optional
from enum import Enum

from pydantic import BaseModel, Field


class RetrievedMemory(BaseModel):
    """A memory returned from retrieval."""
    id: str
    content: str
    timestamp: float
    strength: float
    relevance: float = 0.0


class IntrospectionResult(BaseModel):
    """Result of pet reflecting on a topic."""
    topic: str
    memories: List[RetrievedMemory]
    emotional_impact: Dict[str, float] = Field(default_factory=dict)
    insight: Optional[str] = None


class IntentType(str, Enum):
    CHAT = "chat"
    TASK = "task"


class RoutedIntent(BaseModel):
    """Result of intent classification."""
    intent: IntentType = IntentType.CHAT
    task_spec: Optional[str] = None
    confidence: float = 1.0


class ConversationMessage(BaseModel):
    """A single message in conversation history."""
    role: str
    content: str
    timestamp: float = Field(default_factory=time.time)


class MindResponse(BaseModel):
    """Response from the mind layer."""
    content: str
    memories_used: List[str] = Field(default_factory=list)
    was_task: bool = False
    task_result: Optional[str] = None
