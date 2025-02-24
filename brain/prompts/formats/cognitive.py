"""
Specialized formatter for cognitive processing prompts.
Handles system state, memories, and command interface context.
"""

from typing import Dict, List, Any

from .base import BaseMessageFormat, Message

class CognitiveFormat(BaseMessageFormat):
    """Handles internal cognitive processing prompts."""
    
    def render(self, message: Message, data: Dict[str, Any]) -> str:
        # Get core template from message
        base_content = message.content
        
        if not data:
            return base_content
            
        # Build template data
        template_data = {
            "base_prompt": base_content,
            "state": self._format_state_data(data.get("state", {})),
            "memories": self._format_memory_data(data.get("memories", []))
        }
        
        # Apply cognitive template
        template = self.templates.get("cognitive", "{base_prompt}\n\n{state}\n\n{memories}")
        return template.format(**template_data)
    
    def _format_state_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare state data for templates."""
        return {
            "mood": state.get("mood", {}),
            "behavior": state.get("behavior", {}),
            "needs": state.get("needs", {}),
            "emotional_state": state.get("emotional_state", [])
        }
    
    def _format_memory_data(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare memory data for templates."""
        return [
            {
                "time": mem.get("relative_time", "Recently"),
                "content": mem.get("content", "").strip()
            }
            for mem in memories
        ]
