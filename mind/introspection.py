"""
mind/introspection.py

Periodic self-reflection that makes the pet feel alive between conversations.
Picks a topic, retrieves memories (triggering echoes), and lets the emotional
ripple propagate through the internal state systems.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from event_dispatcher import global_event_dispatcher, Event
from loggers.loggers import BrainLogger

if TYPE_CHECKING:
    from internal.modules.cognition.cognitive_bridge import PetCognitiveBridge
    from .conversation import ConversationManager


class IntrospectionRunner:
    """
    Drives periodic introspection when the pet is idle.

    When no conversation is active, picks a topic (from recent chat or a
    fallback pool) and calls bridge.introspect(), which retrieves memories
    and triggers echo effects that ripple through emotions → mood → behavior.
    """

    IDLE_THRESHOLD: float = 120.0  # seconds since last message to be "idle"
    FALLBACK_TOPICS: list[str] = [
        "things I've been thinking about",
        "recent conversations",
        "how I'm feeling right now",
        "something interesting from earlier",
        "what matters to me",
    ]

    def __init__(
        self,
        bridge: PetCognitiveBridge,
        conversation: ConversationManager,
    ) -> None:
        self.bridge = bridge
        self.conversation = conversation
        self.logger = BrainLogger
        self._topic_index: int = 0

    async def maybe_introspect(self) -> None:
        """Run introspection if the pet is idle (not mid-conversation)."""
        try:
            if not self._is_idle():
                return

            topic = self._pick_topic()
            result = await self.bridge.introspect(topic)

            if result:
                global_event_dispatcher.dispatch_event(Event(
                    "mind:introspection",
                    {
                        "topic": topic,
                        "memory_count": len(result.memories),
                    },
                ))
                self.logger.debug(
                    f"Introspection on '{topic}': {len(result.memories)} memories recalled"
                )
        except Exception as e:
            self.logger.error(f"Introspection failed: {e}")

    def _is_idle(self) -> bool:
        """Check whether the pet is idle (no recent conversation activity)."""
        if self.conversation.is_empty:
            return True

        last_msg = self.conversation.messages[-1]
        return (time.time() - last_msg.timestamp) >= self.IDLE_THRESHOLD

    def _pick_topic(self) -> str:
        """Pick a topic from recent conversation or the fallback pool."""
        if not self.conversation.is_empty:
            # Use the last user message as a reflection seed
            for msg in reversed(self.conversation.messages):
                if msg.role == "user":
                    return msg.content[:100]

        # Fallback: cycle through reflective prompts
        topic = self.FALLBACK_TOPICS[self._topic_index % len(self.FALLBACK_TOPICS)]
        self._topic_index += 1
        return topic
