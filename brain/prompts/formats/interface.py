"""
Specialized formatter for interface-specific prompts.
Handles Discord and user interactions with appropriate context building.
"""

from typing import Dict, List, Any

from .base import BaseMessageFormat, Message

class InterfaceFormat(BaseMessageFormat):
    """Handles external interface prompts."""
    
    def render(self, message: Message, data: Dict[str, Any]) -> str:
        if not data:
            return message.content
            
        interface_type = data.get("interface_type", "user")
        
        # Build template data
        template_data = {
            "base_prompt": message.content,
            "state": self._format_state_data(data.get("state_context", {})),
            "memories": self._format_memory_data(data.get("memories", [])),
            "notifications": self._format_notification_data(data.get("notifications", [])),
            "interaction": self._format_interaction_data(
                interface_type,
                data.get("interaction", {}),
                data.get("history", [])
            )
        }
        
        # Get interface-specific template
        template = self.templates.get(f"{interface_type}_format")
        if not template:
            raise ValueError(f"No template found for interface type: {interface_type}")
            
        return template.format(**template_data)
    
    def _format_state_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Same as CognitiveFormat."""
        return {
            "mood": state.get("mood", {}),
            "behavior": state.get("behavior", {}),
            "needs": state.get("needs", {}),
            "emotional_state": state.get("emotional_state", [])
        }
    
    def _format_memory_data(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Same as CognitiveFormat."""
        return [
            {
                "time": mem.get("relative_time", "Recently"),
                "content": mem.get("content", "").strip()
            }
            for mem in memories
        ]
    
    def _format_notification_data(self, notifications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare notification data for templates."""
        return [
            {
                "type": notif.get("type", "generic"),
                "content": notif.get("content", {}),
                "timestamp": notif.get("timestamp")
            }
            for notif in notifications
        ]
    
    def _format_interaction_data(
        self,
        interface_type: str,
        interaction: Dict[str, Any],
        history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare interaction data for templates."""
        formatted = {
            "type": interface_type,
            "content": interaction.get("content", ""),
            "history": [
                {
                    "timestamp": msg.get("timestamp", ""),
                    "author": msg.get("author", "Unknown"),
                    "content": msg.get("content", "")
                }
                for msg in history
            ]
        }
        
        # Add interface-specific fields
        if interface_type == "discord":
            formatted.update({
                "channel": interaction.get("channel", {}),
                "author": interaction.get("author"),
                "guild": interaction.get("guild")
            })
            
        return formatted