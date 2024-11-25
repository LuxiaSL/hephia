"""
Base environment class for Hephia's tools.

Provides the interface that all tool environments must implement,
ensuring consistent behavior across different tools.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from .terminal_formatter import TerminalFormatter, CommandResponse, EnvironmentHelp

class BaseEnvironment(ABC):
    """Base class for all environments."""
    
    @abstractmethod
    def get_commands(self) -> List[Dict[str, str]]:
        """
        Get available commands for this environment.
        
        Returns:
            List of command dictionaries with name and description
        """
        pass
    
    @abstractmethod
    async def handle_command(self, command: str, context: Dict[str, Any]) -> CommandResponse:
        """
        Handle a command in this environment.
        
        Args:
            command: Command to handle
            context: Current system context
        
        Returns:
            Command response object
        """
        pass
    
    def format_help(self) -> CommandResponse:
        """Format help text for this environment."""
        return TerminalFormatter.format_environment_help(
            EnvironmentHelp(
                self.__class__.__name__,
                self.get_commands(),
                examples=getattr(self, 'help_examples', None),
                tips=getattr(self, 'help_tips', None)
            )
        )