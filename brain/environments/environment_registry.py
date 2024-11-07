"""
Environment registry for Hephia's tool access.

Manages available tool environments and their access patterns,
allowing the cognitive system to interact with both internal
and external tools.
"""

from typing import Dict, Type, Optional
from .base_environment import BaseEnvironment
from .notes import NotesEnvironment
from .search import SearchEnvironment
from .web import WebEnvironment
from api_clients import APIManager

class EnvironmentRegistry:
    """
    Manages and provides access to all available environments.
    """
    
    def __init__(self, api_manager: APIManager):
        """Initialize the environment registry."""
        self.environments: Dict[str, BaseEnvironment] = {}
        self.api_manager = api_manager
        self.setup_environments()
    
    def setup_environments(self):
        """Set up all available environments."""
        # External tools
        self.register_environment("notes", NotesEnvironment())
        self.register_environment("search", SearchEnvironment(self.api_manager))
        self.register_environment("web", WebEnvironment())
        
    
    def register_environment(self, name: str, environment: BaseEnvironment):
        """
        Register a new environment.
        
        Args:
            name: Environment name
            environment: Environment instance
        """
        self.environments[name] = environment
    
    def get_environment(self, name: str) -> Optional[BaseEnvironment]:
        """
        Get environment by name.
        
        Args:
            name: Environment name
        
        Returns:
            BaseEnvironment if found, None otherwise
        """
        return self.environments.get(name.lower())
    
    def get_available_commands(self) -> Dict[str, list]:
        """
        Get all available commands from all environments.
        
        Returns:
            Dict of environment names to their commands
        """
        return {
            name: env.get_commands()
            for name, env in self.environments.items()
        }
    
    async def process_command(self, env_name: str, command: str, context: Dict):
        """
        Process a command in the specified environment.
        
        Args:
            env_name: Environment name
            command: Command to process
            context: Current system context
        
        Returns:
            Command response or error message
        """
        environment = self.get_environment(env_name)
        if not environment:
            return f"Unknown environment: {env_name}"
        
        try:
            return await environment.handle_command(command, context)
        except Exception as e:
            return f"Error in {env_name}: {str(e)}"