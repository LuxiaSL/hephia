"""
commands/model.py - Centralized command modeling for Hephia's terminal interface.

Defines the complete structure of commands from raw LLM input through execution,
supporting a natural terminal-like interface that LLMs can easily understand and use.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum

class CommandParseError(Exception):
    """Raised when command parsing fails."""
    pass

class ParameterType(Enum):
    """Valid parameter types for command arguments."""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"

@dataclass
class Parameter:
    """
    Definition of a command parameter.
    
    Examples:
        notes create <title> <content>
        search query <search_terms>
    """
    name: str
    description: str
    type: ParameterType = ParameterType.STRING
    required: bool = True
    default: Any = None
    examples: List[str] = field(default_factory=list)

@dataclass
class Flag:
    """
    Definition of a command flag.
    
    Examples:
        notes create "title" --tags=important,todo
        search query "term" --limit=10
    """
    name: str
    description: str
    type: ParameterType = ParameterType.STRING
    required: bool = False
    default: Any = None
    examples: List[str] = field(default_factory=list)

@dataclass
class CommandDefinition:
    """
    Complete definition of a command's interface.
    Registered by environments to specify their available commands.
    """
    name: str
    description: str
    parameters: List[Parameter] = field(default_factory=list)
    flags: List[Flag] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    related_commands: List[str] = field(default_factory=list)
    failure_hints: Dict[str, str] = field(default_factory=dict)
    help_text: Optional[str] = None
    category: Optional[str] = None  # For grouping in help displays

@dataclass
class ParsedCommand:
    """
    Represents a command that has been parsed from LLM output.
    Contains the structured interpretation of the command attempt.
    """
    environment: Optional[str]  # None for global commands like 'help'
    action: str
    parameters: List[str]
    flags: Dict[str, str]
    raw_input: str  # Original LLM text for reference/debugging

@dataclass
class CommandValidationError:
    """
    Detailed error information when command validation fails.
    Provides context to help the LLM correct its usage.
    """
    message: str
    suggested_fixes: List[str]
    related_commands: List[str]
    examples: List[str]

@dataclass
class CommandResult:
    """
    Standardized result from command execution.
    Provides rich feedback to guide the LLM's next actions.
    """
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    suggested_commands: List[str] = field(default_factory=list)
    error: Optional[CommandValidationError] = None
    state_changes: Optional[Dict[str, Any]] = None

@dataclass
class EnvironmentCommands:
    """
    Complete command set for an environment.
    Used to register and retrieve available commands.
    """
    environment: str
    description: str
    commands: Dict[str, CommandDefinition]
    category: Optional[str] = None  # For grouping in help displays
    help_text: Optional[str] = None

class GlobalCommands:
    """Constants for global system commands."""
    HELP = "help"
    VERSION = "version"
    
    @classmethod
    def is_global_command(cls, command: str) -> bool:
        """Check if a command is a global system command."""
        return command in [cls.HELP, cls.VERSION]