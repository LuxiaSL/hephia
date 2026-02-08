"""
mind/conversation.py

Conversation history management with sliding window and persistence.
"""

import time
from typing import List, Dict, Optional

from .models import ConversationMessage


class ConversationManager:
    """
    Manages conversation history with sliding window.
    Provides formatted messages for LLM calls and serialization for persistence.
    """

    def __init__(self, max_messages: int = 50):
        self.messages: List[ConversationMessage] = []
        self.max_messages = max_messages
        self._turns_since_last_formation: int = 0

    def add_message(self, role: str, content: str) -> None:
        """Add a message and enforce window limit."""
        self.messages.append(ConversationMessage(
            role=role,
            content=content,
            timestamp=time.time(),
        ))
        if role == "assistant":
            self._turns_since_last_formation += 1

        # Trim oldest messages if over limit
        if len(self.messages) > self.max_messages:
            overflow = len(self.messages) - self.max_messages
            self.messages = self.messages[overflow:]

    def get_history(self, limit: Optional[int] = None) -> List[ConversationMessage]:
        """Get recent messages, optionally limited."""
        if limit is None:
            return list(self.messages)
        return list(self.messages[-limit:])

    def get_formatted_messages(
        self,
        system_prompt: str,
        limit: int = 20,
    ) -> List[Dict[str, str]]:
        """
        Format for LLM API call: [system, ...history].
        Returns messages in the standard role/content dict format.
        """
        formatted: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        recent = self.messages[-limit:] if limit else self.messages
        for msg in recent:
            formatted.append({
                "role": msg.role,
                "content": msg.content,
            })

        return formatted

    def get_recent_excerpt(self, turns: int = 4) -> str:
        """
        Get a plain-text excerpt of recent conversation for memory formation.
        Returns the last N messages formatted as 'Role: content' lines.
        """
        recent = self.messages[-(turns * 2):] if self.messages else []
        lines: List[str] = []
        for msg in recent:
            label = "User" if msg.role == "user" else "You"
            lines.append(f"{label}: {msg.content}")
        return "\n".join(lines)

    @property
    def turns_since_formation(self) -> int:
        """Number of assistant turns since last memory formation attempt."""
        return self._turns_since_last_formation

    def reset_formation_counter(self) -> None:
        """Reset the turns-since-formation counter after a formation attempt."""
        self._turns_since_last_formation = 0

    @property
    def is_empty(self) -> bool:
        """Whether there are any messages."""
        return len(self.messages) == 0

    @property
    def message_count(self) -> int:
        """Total number of messages."""
        return len(self.messages)

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()
        self._turns_since_last_formation = 0

    def to_state(self) -> List[Dict[str, str]]:
        """Serialize for persistence (compatible with StateBridge.brain_state)."""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ]

    def from_state(self, state: List[Dict[str, str]]) -> None:
        """Restore from persisted state."""
        self.messages.clear()
        self._turns_since_last_formation = 0

        if not isinstance(state, list):
            return

        for entry in state:
            if isinstance(entry, dict) and "role" in entry and "content" in entry:
                self.messages.append(ConversationMessage(
                    role=entry["role"],
                    content=entry["content"],
                    timestamp=time.time(),
                ))
