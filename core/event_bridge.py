"""
Simplified event bridge for MVP state updates.
Handles core state changes between internal systems and external interfaces.
"""

from typing import Dict, Any
from event_dispatcher import global_event_dispatcher, Event

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
        global_event_dispatcher.add_listener("need:changed", self.handle_state_change)
        global_event_dispatcher.add_listener("behavior:changed", self.handle_state_change)
        global_event_dispatcher.add_listener("mood:changed", self.handle_state_change)
        global_event_dispatcher.add_listener("emotion:new", self.handle_state_change)
    
    async def handle_state_change(self, event: Event):
        """
        Handle core state changes, updating state bridge as needed.
        
        Args:
            event: State change event from internal systems
        """
        # Update state bridge with new state information
        await self.state_bridge.update_state(internal_state={
            "type": event.event_type,
            "data": event.data
        })