"""
brain/core/command_handler.py

Handles all command processing logic for the terminal environment system.
Acts as an abstraction layer between interfaces and environment commands,
managing preprocessing, validation, execution and result formatting.
"""

from typing import Dict, Any, Tuple, Optional

from brain.commands.preprocessor import CommandPreprocessor
from brain.environments.environment_registry import EnvironmentRegistry
from brain.environments.terminal_formatter import TerminalFormatter
from brain.commands.model import (
    CommandResult,
    ParsedCommand,
    CommandValidationError,
    GlobalCommands
)
from config import Config
from loggers import BrainLogger
from api_clients import APIManager
from core.state_bridge import StateBridge

class CommandHandler:
    def __init__(
        self,
        api_manager: APIManager,
        environment_registry: EnvironmentRegistry,
        state_bridge: StateBridge
    ):
        self.api = api_manager
        self.environment_registry = environment_registry
        self.state_bridge = state_bridge
        self.command_preprocessor = CommandPreprocessor(self.api)
        self.last_environment = None

    async def preprocess_command(
        self,
        llm_response: str
    ) -> Tuple[Optional[ParsedCommand], Optional[str]]:
        """
        Preprocess and validate an LLM response into a command.
        
        Args:
            llm_response: Raw LLM output to process
            
        Returns:
            Tuple of (parsed command or None, error message or None)
        """
        try:
            command, error = await self.command_preprocessor.preprocess_command(
                llm_response,
                self.environment_registry.get_available_commands()
            )
            
            if error:
                return None, TerminalFormatter.format_error(error)
                
            return command, None
            
        except Exception as e:
            error_msg = f"Command preprocessing failed: {str(e)}"
            BrainLogger.error(error_msg)
            return None, error_msg

    async def execute_command(
        self,
        command: ParsedCommand,
        context: Optional[Dict[str, Any]] = None
    ) -> CommandResult:
        """
        Execute a parsed command.
        
        Args:
            command: Validated command structure
            context: Optional state context for command execution
        """
        try:
            # Get fresh state context if none provided
            if context is None:
                context = await self.state_bridge.get_api_context()

            # Handle global commands
            if not command.environment:
                return await self._handle_global_command(command)

            # Handle environment commands
            result = await self._handle_environment_command(command, context)
            
            # Update last environment tracking
            if result.success:
                self.last_environment = command.environment
                
            return result

        except Exception as e:
            error_msg = f"Command execution failed: {str(e)}"
            BrainLogger.error(error_msg)
            return CommandResult(
                success=False,
                message=error_msg,
                suggested_commands=["help"],
                error=CommandValidationError(
                    message=error_msg,
                    suggested_fixes=["Check command syntax", "Verify system state"],
                    related_commands=["help"],
                    examples=["help"]
                )
            )

    async def _handle_global_command(self, command: ParsedCommand) -> CommandResult:
        """Handle global system commands."""
        if command.action == GlobalCommands.HELP:
            return self.environment_registry.format_global_help()
            
        if command.action == GlobalCommands.VERSION:
            return CommandResult(
                success=True,
                message=f"Hephia OS v{Config.VERSION}",
                suggested_commands=["help"]
            )
            
        return CommandResult(
            success=False,
            message=f"Unknown global command: {command.action}",
            suggested_commands=["help"],
            error=CommandValidationError(
                message=f"Command '{command.action}' is not a valid global command",
                suggested_fixes=["Use 'help' to see available commands"],
                related_commands=["help", "version"],
                examples=["help", "version"]
            )
        )

    async def _handle_environment_command(
        self,
        command: ParsedCommand,
        context: Dict[str, Any]
    ) -> CommandResult:
        """Handle environment-specific commands."""
        # Get environment handler
        environment = self.environment_registry.get_environment(command.environment)
        if not environment:
            return CommandResult(
                success=False,
                message=f"Unknown environment: {command.environment}",
                suggested_commands=["help"],
                error=CommandValidationError(
                    message=f"Environment '{command.environment}' not found",
                    suggested_fixes=["Check environment spelling", "Use 'help' to see available environments"],
                    related_commands=["help"],
                    examples=[f"{env} help" for env in self.environment_registry.get_available_commands().keys()]
                )
            )

        try:
            # Execute command in environment
            return await environment.handle_command(command, context)
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error in {command.environment}: {str(e)}",
                suggested_commands=[f"{command.environment} help"],
                error=CommandValidationError(
                    message=str(e),
                    suggested_fixes=["Check command syntax", "Verify environment state"],
                    related_commands=[f"{command.environment} help"],
                    examples=[f"{command.environment} help"]
                )
            )