"""
mind/task_router.py

Intent classification — determines whether user input is chat or a task request.
Heuristic-first approach: pattern matching for obvious cases, LLM fallback for ambiguous ones.
"""

from __future__ import annotations

import re
from typing import List, Dict, Optional, TYPE_CHECKING

from .models import IntentType, RoutedIntent

if TYPE_CHECKING:
    from .pet_model import PetModel


# Patterns that strongly suggest a task request
_TASK_PATTERNS = [
    r"\b(search|look up|find|google|calculate|compute)\b",
    r"\b(write|generate|create|draft|compose)\s+(a |an |me |the )?(code|script|function|program|email|letter|essay|report|summary)",
    r"\b(translate|convert|format|parse|analyze)\b",
    r"\b(explain|summarize|describe)\s+(how|what|why|the)\b",
    r"\b(help me with|assist with|can you do)\b",
]

# Patterns that strongly suggest chat
_CHAT_PATTERNS = [
    r"^(hi|hey|hello|yo|sup|good morning|good evening|howdy)\b",
    r"^(how are you|what's up|how do you feel|how you doing)",
    r"^(thanks|thank you|bye|goodbye|see you|good night)",
    r"\b(love you|miss you|like you)\b",
]

_TASK_RE = [re.compile(p, re.IGNORECASE) for p in _TASK_PATTERNS]
_CHAT_RE = [re.compile(p, re.IGNORECASE) for p in _CHAT_PATTERNS]


class TaskRouter:
    """Classifies user intent as chat or task."""

    def __init__(self, pet_model: PetModel) -> None:
        self.pet_model = pet_model

    async def classify(
        self,
        user_message: str,
        conversation_context: Optional[List[Dict[str, str]]] = None,
    ) -> RoutedIntent:
        """
        Determine if input is chat or a task request.
        Uses heuristics first, falls back to LLM classification for ambiguous cases.
        """
        # Try heuristic classification first
        heuristic = self._heuristic_classify(user_message)
        if heuristic is not None:
            return heuristic

        # Short messages default to chat
        if len(user_message.split()) <= 3:
            return RoutedIntent(intent=IntentType.CHAT, confidence=0.7)

        # Ambiguous — use LLM classification
        try:
            result = await self.pet_model.classify_intent(user_message)
            if result == "TASK":
                return RoutedIntent(
                    intent=IntentType.TASK,
                    task_spec=user_message,
                    confidence=0.8,
                )
            return RoutedIntent(intent=IntentType.CHAT, confidence=0.8)
        except Exception:
            # Default to chat on classification failure
            return RoutedIntent(intent=IntentType.CHAT, confidence=0.5)

    def _heuristic_classify(self, message: str) -> Optional[RoutedIntent]:
        """Fast pattern-based classification for obvious cases."""
        # Check chat patterns first (greetings, etc.)
        for pattern in _CHAT_RE:
            if pattern.search(message):
                return RoutedIntent(intent=IntentType.CHAT, confidence=0.95)

        # Check task patterns
        for pattern in _TASK_RE:
            if pattern.search(message):
                return RoutedIntent(
                    intent=IntentType.TASK,
                    task_spec=message,
                    confidence=0.9,
                )

        return None
