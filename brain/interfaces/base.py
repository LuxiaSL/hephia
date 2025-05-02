"""
brain/interfaces/base.py

Base classes for cognitive interfaces, providing common functionality
for context formatting, notification management, and cognitive continuity.
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, List, Optional, Any
from loggers import BrainLogger
from event_dispatcher import global_event_dispatcher, Event
from brain.utils.tracer import brain_trace
from brain.cognition.notification import Notification, NotificationManager, NotificationInterface
from brain.cognition.memory.significance import MemoryData
from brain.environments.terminal_formatter import TerminalFormatter

from core.state_bridge import StateBridge
from internal.modules.cognition.cognitive_bridge import CognitiveBridge

class CognitiveInterface(NotificationInterface, ABC):
    def __init__(
        self, 
        interface_id: str,
        state_bridge: StateBridge,  
        cognitive_bridge: CognitiveBridge,
        notification_manager: NotificationManager,
    ):
        super().__init__(interface_id, notification_manager)
        self.state_bridge = state_bridge
        self.cognitive_bridge = cognitive_bridge

    @abstractmethod
    async def process_interaction(self, content: Any) -> Any:
        """Process an interaction through this interface."""
        pass

    @abstractmethod
    async def format_memory_context(
        self,
        content: Any,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format context specifically for memory formation.
        Uses same context as regular interactions but formatted for memory creation.
        """
        pass

    @abstractmethod
    async def get_fallback_memory(self, memory_data: MemoryData) -> Optional[str]:
        """Get a fallback memory for this interface."""
        pass
    
    @brain_trace
    async def get_cognitive_context(self, metadata: Optional[Dict[str, Any]] = None) -> AsyncGenerator[tuple[str, Any], None]:
        """Get formatted cognitive context including state and notifications, yielding each piece."""
        brain_trace.context.memory("Getting relevant memories")
        memories = await self.get_relevant_memories(metadata=metadata) or []
        BrainLogger.debug(f"[{self.interface_id}] Got memories: {memories}")

        brain_trace.context.state("Retrieving API context")
        state = await self.state_bridge.get_api_context() or {}
        BrainLogger.debug(f"[{self.interface_id}] Got API context: {state}")

        brain_trace.context.format("Formatting cognitive context")
        formatted_context = TerminalFormatter.format_context_summary(state, memories) or "No context available"
        yield ("formatted_context", formatted_context)
        
        brain_trace.context.notifications("Getting updates from other interfaces")
        other_updates = await self.notification_manager.get_updates_for_interface(self.interface_id)
        BrainLogger.info(f"[{self.interface_id}] Got updates from other interfaces: {other_updates}")
        yield ("updates", other_updates)
    
    @abstractmethod
    async def get_relevant_memories(self, metadata: Optional[Dict[str, any]] = None) -> List[Dict[str, Any]]:
        """Retrieve memories relevant to this interface's current context."""
        pass

    async def announce_cognitive_context(self, raw, notification: Notification) -> None:
        """Notify event system about new cognitive context."""
        try:
            # Get our own summary to announce
            summary = await self.notification_manager._summary_formatters[self.interface_id]([notification])  # Pass relevant notifications
            
            await global_event_dispatcher.dispatch_event_async(Event(
                "cognitive:context_update",
                {
                    "source": self.interface_id,
                    "raw_state": raw,
                    "processed_state": summary
                }
            ))
            BrainLogger.info(f"[{self.interface_id}] Context announcement complete")
        except Exception as e:
            BrainLogger.error(f"Error announcing cognitive context for {self.interface_id}: {e}")