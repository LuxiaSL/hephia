"""
Simplified event bridge for MVP state updates.
Handles core state changes between internal systems and external interfaces.
"""

from typing import Dict, Any
from event_dispatcher import global_event_dispatcher, Event
import asyncio

class EventBridge:
    """
    Manages essential event flow between internal systems and external interfaces.
    Focused on state updates and basic command processing for MVP.
    """
    
    def __init__(self, state_bridge):
        """Initialize event bridge with state bridge reference."""
        self.state_bridge = state_bridge
        self.setup_listeners()
    
    def setup_listeners(self):
        """Set up core event listeners."""
        # Core state change events
        global_event_dispatcher.add_listener(
            "need:changed", 
            lambda event: asyncio.create_task(self.handle_state_change(event))
        )
        global_event_dispatcher.add_listener(
            "behavior:changed",
            lambda event: asyncio.create_task(self.handle_state_change(event))
        )
        global_event_dispatcher.add_listener(
            "mood:changed",
            lambda event: asyncio.create_task(self.handle_state_change(event))
        )
        global_event_dispatcher.add_listener(
            "emotion:new",
            lambda event: asyncio.create_task(self.handle_state_change(event))
        )
        
        global_event_dispatcher.add_listener(
            "emotion:update",
            lambda event: asyncio.create_task(self.handle_emotion_update(event))
        )

        global_event_dispatcher.add_listener(
            "*:echo",
            lambda event: asyncio.create_task(self.handle_state_change(event))
        )
        
    async def handle_emotion_update(self, event: Event):
        """Handle emotion updates with backoff and rate limiting."""
        # Try to acquire lock - if locked, discard the update
        if self._emotion_update_lock.locked():
            return
            
        async with self._emotion_update_lock:
            await asyncio.sleep(0.1)  # 100ms backoff
            await self.handle_state_change(event)

    async def handle_state_change(self, event: Event):
        """
        Handle core state changes, updating state bridge as needed.
        
        Args:
            event: State change event from internal systems
        """
        # Update state bridge with new state information
        await self.state_bridge.update_state()