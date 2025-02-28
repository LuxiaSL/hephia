"""
Specialized formatter for memory formation prompts.
Uses templated prompts while maintaining consistent context formatting.
"""

from typing import Dict, Any

from .base import BaseMessageFormat, Message

class MemoryFormat(BaseMessageFormat):
    """Handles memory formation prompts."""
    
    def render(self, message: Message, data: Dict[str, Any]) -> str:
        if not data:
            return message.content
            
        interface_type = data.get("interface_type", "exo")
        
        # Build template data
        template_data = {
            "base_prompt": message.content,
            "state": self._format_state_data(data.get("state", {})),
            "interaction": self._format_interaction_data(
                interface_type,
                data.get("interaction", {})
            )
        }
        
        # Get memory formation template for this interface
        template = self.templates.get(f"memory_{interface_type}")
        if not template:
            raise ValueError(f"No memory template for interface type: {interface_type}")
            
        return template.format(**template_data)
    
    def _format_state_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Same as other formatters."""
        return {
            "mood": state.get("mood", {}),
            "behavior": state.get("behavior", {}),
            "needs": state.get("needs", {}),
            "emotional_state": state.get("emotional_state", [])
        }
    
    def _format_interaction_data(
        self,
        interface_type: str,
        interaction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare interaction data for memory templates."""
        formatted = {
            "type": interface_type,
            "content": interaction.get("content", ""),
            "response": interaction.get("response", "")
        }
        
        # Add interface-specific fields
        if interface_type == "discord":
            formatted.update({
                "channel": interaction.get("channel", {}),
                "author": interaction.get("author"),
                "guild": interaction.get("guild"),
                "history": interaction.get("history", [])
            })
            
        return formatted