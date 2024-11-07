"""
Core exo processor for MVP implementation.
Handles basic context retrieval and command processing for LLM interaction.
"""

from typing import Dict, Any, List
from brain.environments.environment_registry import EnvironmentRegistry
from brain.environments.ui_formatter import UIFormatter

class ExoProcessor:
    """
    Manages LLM interactions and command processing for MVP implementation.
    Focuses on context retrieval and basic command handling.
    """
    
    def __init__(self, state_bridge):
        """
        Initialize the exo processor.
        
        Args:
            state_bridge: Bridge to access system state
        """
        self.state_bridge = state_bridge
        self.environment_registry = EnvironmentRegistry()
        
    async def get_pet_context(self) -> Dict[str, Any]:
        """
        Retrieve current pet context from state bridge.
        
        Returns:
            dict: Current pet context including mood, needs, behavior
        """
        state = await self.state_bridge.get_current_state()
        return state["pet_state"]
        
    async def process_command(self, command: str, context: Dict[str, Any]) -> str:
        """
        Process a command with current pet context.
        
        Args:
            command: Command to process
            context: Current pet context
        
        Returns:
            str: Formatted command response
        """
        # Split into environment and action
        parts = command.split(" ", 1)
        if len(parts) < 2:
            return self._format_help_response(context)
            
        env_name, action = parts
        
        # Process through appropriate environment
        environment = self.environment_registry.get_environment(env_name)
        if not environment:
            return UIFormatter.format_error(f"Unknown environment: {env_name}")
            
        try:
            response = await environment.handle_command(action, context)
            return UIFormatter.format_command_response(
                title=f"{env_name.capitalize()} Command",
                content=response,
                context=self._format_context(context)
            )
        except Exception as e:
            return UIFormatter.format_error(str(e))
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format pet context for display."""
        return f"""Current State:
Mood: {context.get('mood', {}).get('name', 'Unknown')}
Needs: {', '.join(f"{k}: {v}" for k, v in context.get('needs', {}).items())}
Behavior: {context.get('current_behavior', 'Unknown')}"""
    
    def _format_help_response(self, context: Dict[str, Any]) -> str:
        """Format help response with available environments."""
        environments = self.environment_registry.get_available_commands()
        
        help_text = "Available Commands:\n"
        for env_name, commands in environments.items():
            help_text += f"\n{env_name}:\n"
            for cmd in commands:
                help_text += f"  {cmd['name']} - {cmd['description']}\n"
        
        return UIFormatter.format_command_response(
            title="Help",
            content=help_text,
            context=self._format_context(context)
        )