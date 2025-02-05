"""
base_environment.py - Base class for all Hephia environments.

Provides structured command registration and handling while maintaining
the terminal-like interface that makes it easy for LLMs to interact with.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

from brain.commands.model import (
    CommandDefinition,
    ParameterType,
    ParsedCommand,
    CommandResult,
    EnvironmentCommands,
    CommandValidationError
)

class BaseEnvironment(ABC):
    """
    Base class for all environments.
    Provides command registration and structured command handling.
    """
    
    def __init__(self):
        """Initialize environment with empty command registry."""
        self.name: str = self.__class__.__name__.lower().replace('environment', '')
        self.description: str = self.__doc__ or ""
        self.commands: Dict[str, CommandDefinition] = {}
        self.help_text: Optional[str] = None
        self._register_commands()

    @abstractmethod
    def _register_commands(self) -> None:
        """
        Register all commands available in this environment.
        
        Example implementation:
        ```python
        self.register_command(
            CommandDefinition(
                name="query",
                description="Search and summarize information",
                parameters=[
                    Parameter(
                        name="search_terms",
                        description="Terms to search for",
                        required=True,
                        examples=["current weather in London"]
                    )
                ],
                flags=[
                    Flag(
                        name="limit",
                        description="Maximum results to return",
                        type=ParameterType.INTEGER,
                        default=5
                    )
                ],
                examples=[
                    'search query "Python programming"',
                    'search query "Weather forecast" --limit=3'
                ]
            )
        )
        ```
        """
        pass

    def register_command(self, command: CommandDefinition) -> None:
        """Register a single command."""
        self.commands[command.name] = command

    def get_environment_info(self) -> EnvironmentCommands:
        """Get complete environment command information."""
        return EnvironmentCommands(
            environment=self.name,
            description=self.description,
            commands=self.commands,
            help_text=self.help_text
        )

    async def handle_command(
        self, 
        command: ParsedCommand,
        context: Dict[str, Any]
    ) -> CommandResult:
        """
        Handle a parsed command with validation.
        
        Args:
            command: Preprocessed and parsed command
            context: Current system context
            
        Returns:
            Command execution result with feedback
        """
        # Get command definition
        cmd_def = self.commands.get(command.action)
        if not cmd_def:
            return CommandResult(
                success=False,
                message=f"Unknown command '{command.action}'",
                suggested_commands=self._get_similar_commands(command.action),
                error=CommandValidationError(
                    message=f"Unknown command '{command.action}'",
                    suggested_fixes=["Check command spelling", "Use help to see available commands"],
                    related_commands=self._get_similar_commands(command.action),
                    examples=self._get_command_examples()
                )
            )

        # Validate parameters and flags
        try:
            validated_params = self._validate_parameters(cmd_def, command.parameters)
            validated_flags = self._validate_flags(cmd_def, command.flags)
        except ValueError as e:
            return CommandResult(
                success=False,
                message=str(e),
                suggested_commands=[
                    f"{self.name} {cmd}" for cmd in cmd_def.related_commands
                ],
                error=CommandValidationError(
                    message=str(e),
                    suggested_fixes=self._generate_parameter_hints(cmd_def),
                    related_commands=cmd_def.related_commands,
                    examples=cmd_def.examples
                )
            )

        try:
            # Execute command-specific logic
            result = await self._execute_command(
                command.action, 
                validated_params, 
                validated_flags, 
                context
            )
            
            # Enhance result with related commands if not provided
            if not result.suggested_commands and cmd_def.related_commands:
                result.suggested_commands = [
                    f"{self.name} {cmd}" for cmd in cmd_def.related_commands
                ]
            
            return result
            
        except Exception as e:
            # Get specific failure hint if available
            hint = cmd_def.failure_hints.get(str(e), str(e))
            return CommandResult(
                success=False,
                message=f"Command failed: {hint}",
                suggested_commands=self._get_recovery_commands(command.action),
                error=CommandValidationError(
                    message=str(e),
                    suggested_fixes=[hint] if hint else [],
                    related_commands=self._get_recovery_commands(command.action),
                    examples=cmd_def.examples
                )
            )

    def _validate_parameters(
        self, 
        command: CommandDefinition, 
        params: List[str]
    ) -> List[Any]:
        """Validate and convert parameters according to command spec."""
        required_params = [p for p in command.parameters if p.required]
        if len(params) < len(required_params):
            param_names = [p.name for p in required_params]
            raise ValueError(
                f"Command '{command.name}' requires parameters: {', '.join(param_names)}"
            )

        validated = []
        for i, param_spec in enumerate(command.parameters):
            if i < len(params):
                try:
                    value = self._convert_parameter(params[i], param_spec.type)
                    validated.append(value)
                except ValueError:
                    raise ValueError(
                        f"Parameter '{param_spec.name}' must be of type {param_spec.type.value}"
                    )
            elif param_spec.default is not None:
                validated.append(param_spec.default)
            elif param_spec.required:
                raise ValueError(f"Missing required parameter '{param_spec.name}'")

        return validated

    def _validate_flags(
        self, 
        command: CommandDefinition, 
        flags: Dict[str, str]
    ) -> Dict[str, Any]:
        """Validate and convert flags according to command spec."""
        validated = {}
        flag_specs = {f.name: f for f in command.flags}

        # Validate provided flags
        for name, value in flags.items():
            if name not in flag_specs:
                raise ValueError(
                    f"Unknown flag '--{name}' for command '{command.name}'"
                )
            
            spec = flag_specs[name]
            try:
                validated[name] = self._convert_parameter(value, spec.type)
            except ValueError:
                raise ValueError(
                    f"Flag '--{name}' must be of type {spec.type.value}"
                )

        # Add required flags with defaults
        for spec in command.flags:
            if spec.required and spec.name not in validated:
                if spec.default is not None:
                    validated[spec.name] = spec.default
                else:
                    raise ValueError(f"Missing required flag '--{spec.name}'")

        return validated

    def _convert_parameter(self, value: str, param_type: ParameterType) -> Any:
        """Convert parameter string to specified type."""
        converters = {
            ParameterType.STRING: str,
            ParameterType.NUMBER: float,
            ParameterType.INTEGER: int,
            ParameterType.BOOLEAN: lambda x: x.lower() == "true"
        }
        return converters[param_type](value)

    def _get_similar_commands(self, action: str) -> List[str]:
        """Get list of similar commands for suggestions."""
        return [
            name for name in self.commands.keys()
            if name.startswith(action[:2]) or
            any(word in name for word in action.split("_"))
        ]

    def _get_recovery_commands(self, action: str) -> List[str]:
        """Get suggested recovery commands after a failure."""
        command = self.commands.get(action)
        if not command:
            return []
        return [f"{self.name} {cmd}" for cmd in command.related_commands]

    def _get_command_examples(self) -> List[str]:
        """Get a list of command examples from all commands."""
        examples = []
        for cmd in self.commands.values():
            examples.extend(cmd.examples)
        return examples[:5]  # Return top 5 examples

    def _generate_parameter_hints(self, command: CommandDefinition) -> List[str]:
        """Generate helpful hints about command parameters."""
        hints = []
        
        # Required parameters
        required = [p for p in command.parameters if p.required]
        if required:
            hints.append(
                f"Required parameters: {', '.join(f'<{p.name}>' for p in required)}"
            )
            
        # Available flags
        if command.flags:
            hints.append(
                f"Available flags: {', '.join(f'--{f.name}' for f in command.flags)}"
            )
            
        # Format example
        param_str = " ".join(
            f"<{p.name}>" if p.required else f"[{p.name}]"
            for p in command.parameters
        )
        flag_str = " ".join(
            f"[--{f.name}]" for f in command.flags
        )
        hints.append(f"Format: {self.name} {command.name} {param_str} {flag_str}")
        
        return hints

    @abstractmethod
    async def _execute_command(
        self,
        action: str,
        params: List[Any],
        flags: Dict[str, Any],
        context: Dict[str, Any]
    ) -> CommandResult:
        """
        Execute a validated command.
        
        Args:
            action: Command action to perform
            params: Validated parameters
            flags: Validated flags
            context: Current system context
        """
        pass