"""
brain/interfaces/action.py - Action interface for Hephia's action system.

Handles notification generation and summaries for actions executed by the system.
Maintains first-person perspective for cognitive continuity.
"""

from typing import List
from datetime import datetime
from brain.cognition.notification import Notification, NotificationManager
from brain.interfaces.base import CognitiveInterface
from core.state_bridge import StateBridge
from internal.modules.cognition.cognitive_bridge import CognitiveBridge


class ActionInterface(CognitiveInterface):
    """
    Interface for action system notifications.
    Provides first-person summaries of actions taken to maintain cognitive continuity.
    """
    
    def __init__(
        self,
        state_bridge: StateBridge,
        cognitive_bridge: CognitiveBridge,
        notification_manager: NotificationManager
    ):
        super().__init__("action", state_bridge, cognitive_bridge, notification_manager)

    async def _generate_summary(self, notifications: List[Notification]) -> str:
        """
        Generate first-person summaries of actions taken.
        
        Formats action effects and messages in a natural, embodied style that
        maintains cognitive continuity with other interfaces.
        """
        formatted = []
        
        for notif in notifications:
            content = notif.content
            action_name = content.get('action', '').replace('_', ' ')
            message = content.get('message', '')
            state_changes = content.get('state_changes', {})
            result = content.get('result', {})
            
            # Format base action in first person
            summary = f"The user performed '{action_name}' and helped me!"
            
            # Add user message if provided
            if message:
                summary += f" They also mentioned that: ({message})"
            
            # Format significant state changes
            if state_changes:
                changes = []
                for key, value in state_changes.items():
                    # Handle numerical changes
                    if isinstance(value, (int, float)):
                        if abs(value) > 0.1:  # Only show significant changes
                            changes.append(f"my {key} changed by {value:+.1f}")
                    else:
                        changes.append(f"my {key} became {value}")
                
                if changes:
                    summary += f"\n- Effects: {', '.join(changes)}"
            
            # Add timestamp context
            if 'timestamp' in content:
                try:
                    timestamp = content['timestamp']
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp)
                    time_str = timestamp.strftime("%H:%M:%S")
                    summary = f"[{time_str}] {summary}"
                except (ValueError, TypeError):
                    pass  # Skip timestamp if format is invalid
            
            formatted.append(summary)
        
        # Return most recent summaries, maintaining cognitive recency
        recent_summaries = formatted[-5:]  # Keep last 5 actions
        if len(formatted) > 5:
            summary = "Recent actions:\n" + "\n".join(recent_summaries)
        else:
            summary = "\n".join(recent_summaries)
            
        return summary
    
    async def process_interaction(self, content):
        pass

    async def format_cognitive_context(self, state, memories):
        pass

    async def format_memory_context(self, content, state, metadata=None):
        pass

    async def get_relevant_memories(self):
        pass