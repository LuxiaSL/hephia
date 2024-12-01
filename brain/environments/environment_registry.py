"""
Environment registry for Hephia's tool access.

Manages available tool environments and their access patterns,
allowing the cognitive system to interact with both internal
and external tools.
"""

from typing import Dict, List, Any, Optional
from .base_environment import BaseEnvironment, CommandResult
from .notes import NotesEnvironment
from .search import SearchEnvironment
from .web import WebEnvironment
from api_clients import APIManager
from brain.commands.model import (
    CommandDefinition,
    EnvironmentCommands,
    CommandValidationError
)

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
        """Register a new environment."""
        self.environments[name] = environment
    
    def get_environment(self, name: str) -> Optional[BaseEnvironment]:
        """Get environment by name."""
        return self.environments.get(name.lower())
    
    def get_available_commands(self) -> Dict[str, List[Dict[str, str]]]:
        """Get complete command information for all environments."""
        return {
            name: env.get_environment_info()
            for name, env in self.environments.items()
        }

    def format_global_help(self) -> CommandResult:
        """Format comprehensive help for all environments."""
        help_sections = []
        
        # Build help text from environment info
        for env_name, env in self.environments.items():
            env_info = env.get_environment_info()
            
            # Add environment header
            help_sections.append(f"\n{env_name.upper()} COMMANDS:")
            help_sections.append(f"  {env_info.description}\n")
            
            # Group commands by category
            categorized = {}
            for cmd in env_info.commands.values():
                category = cmd.category or "General"
                if category not in categorized:
                    categorized[category] = []
                categorized[category].append(cmd)
            
            # Format each category
            for category, commands in categorized.items():
                help_sections.append(f"  {category}:")
                for cmd in commands:
                    # Format command signature
                    params = " ".join(
                        f"<{p.name}>" if p.required else f"[{p.name}]"
                        for p in cmd.parameters
                    )
                    flags = " ".join(f"[--{f.name}]" for f in cmd.flags)
                    signature = f"    {env_name} {cmd.name} {params} {flags}".strip()
                    
                    # Add command details
                    help_sections.append(signature)
                    help_sections.append(f"      {cmd.description}")
                    if cmd.examples:
                        help_sections.append("      Examples:")
                        help_sections.extend(
                            f"        {ex}" for ex in cmd.examples[:2]
                        )
                help_sections.append("")  # Spacing between categories
        
        return CommandResult(
            success=True,
            message="\n".join(help_sections),
            suggested_commands=[
                f"{env} help" for env in self.environments.keys()
            ]
        )