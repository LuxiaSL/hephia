"""
Base environment class for Hephia's tools.

Provides the interface that all tool environments must implement,
ensuring consistent behavior across different tools.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from .ui_formatter import UIFormatter

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
    async def handle_command(self, command: str, context: Dict[str, Any]) -> str:
        """
        Handle a command in this environment.
        
        Args:
            command: Command to handle
            context: Current system context
        
        Returns:
            Command response
        """
        pass
    
    def format_help(self) -> Dict[str, str]:
        """Format help text for this environment."""
        return UIFormatter.format_environment_help(
            self.__class__.__name__,
            self.get_commands(),
            examples=getattr(self, 'help_examples', None),
            tips=getattr(self, 'help_tips', None)
        )