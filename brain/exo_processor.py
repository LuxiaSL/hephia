"""
exo_processor.py - Core exo-LLM loop management.

Manages the continuous cognitive loop between LLM, command processing,
and environment interactions. Maintains conversation state and ensures
smooth flow between components.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import asyncio
import os
import json

from brain.commands.preprocessor import CommandPreprocessor
from brain.environments.environment_registry import EnvironmentRegistry
from brain.environments.terminal_formatter import TerminalFormatter
from loggers import BrainLogger
from brain.commands.model import (
    CommandResult,
    ParsedCommand,
    CommandValidationError,
    GlobalCommands
)
from config import Config

class ExoProcessor:
    def __init__(self, api_manager, state_bridge, environment_registry):
        self.api = api_manager
        self.state_bridge = state_bridge
        self.environment_registry = environment_registry
        self.command_preprocessor = CommandPreprocessor(self.api)
        self.conversation_history = []
        self.exo_lock = asyncio.Lock()
        self.last_successful_turn = None

    async def initialize(self):
        """Handle async initialization tasks."""
        initial_state = await self.state_bridge.get_current_state()
        formatted_state = TerminalFormatter.format_context_summary(initial_state)
        
        welcome_message = {
            "role": "system",
            "content": Config.SYSTEM_PROMPT
        }
        
        initial_message = {
            "role": "user",
            "content": f"{TerminalFormatter.format_welcome()}\n\n{formatted_state}"
        }
        
        self.conversation_history = [welcome_message, initial_message]

    async def process_turn(self) -> Optional[str]:
        """
        Process one turn of the exo-LLM loop.
        
        Each turn follows the flow:
        1. Get LLM response
        2. Parse into command
        3. Execute command
        4. Format response
        5. Update conversation state
        """
        BrainLogger.log_turn_start()

        if self.last_successful_turn and \
           (datetime.now() - self.last_successful_turn).total_seconds() < Config.EXO_MIN_INTERVAL:
            BrainLogger.log_turn_end(False, "Too soon since last completion")
            return None

        try:
            async with asyncio.timeout(Config.EXO_TIMEOUT):
                if self.exo_lock.locked():
                    BrainLogger.log_turn_end(False, "Lock held")
                    return None
                    
                async with self.exo_lock:
                    try:
                        # Get LLM response
                        response = await self._get_llm_response()

                        llm_response = response["choices"][0]["message"]["content"]
                        BrainLogger.log_llm_exchange(self.conversation_history, llm_response)

                        # Process into command
                        command, error = await self.command_preprocessor.preprocess_command(
                            llm_response,
                            self.environment_registry.get_available_commands()
                        )

                        BrainLogger.log_command_processing(
                            response,
                            command.raw_input if command else None,
                            str(error) if error else None
                        )

                        # Handle preprocessing errors
                        if not command:
                            error_message = TerminalFormatter.format_error(error)
                            self._add_to_history("assistant", llm_response)
                            self._add_to_history("user", error_message)
                            return error_message
                        
                        self._add_to_history("assistant", llm_response)
                        
                        # Execute command
                        current_state = await self.state_bridge.get_current_state()
                        result = await self._execute_command(command, current_state)

                        # Format response
                        formatted_response = TerminalFormatter.format_command_result(
                            result,
                            current_state
                        )

                        # Update conversation
                        self._add_to_history("user", formatted_response)
                        self.last_successful_turn = datetime.now()
                        
                        BrainLogger.log_turn_end(True)
                        return formatted_response
                    
                    except asyncio.TimeoutError:
                        BrainLogger.log_turn_end(False, "Exo loop run timeout")
                        return None
                    except Exception as e:
                        BrainLogger.log_turn_end(False, str(e))
                        return None
        except asyncio.TimeoutError:
            BrainLogger.log_turn_end(False, "Exo loop lock timeout")
            return None
        except Exception as e:
            BrainLogger.log_turn_end(False, f"Fatal error: {str(e)}")
            return None

    async def _get_llm_response(self) -> Dict:
        """Get next action from LLM."""
        return await self.api.openpipe.create_completion(
            model=os.getenv("OPENPIPE_MODEL"),
            messages=self.conversation_history,
            temperature=Config.EXO_TEMPERATURE
        )

    async def _execute_command(
        self,
        command: ParsedCommand,
        context: Dict[str, Any]
    ) -> CommandResult:
        """
        Execute a parsed command.
        
        Args:
            command: Validated command structure
            context: Current system state
        """
        # Handle global commands
        if not command.environment:
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
                suggested_commands=["help"]
            )
        
        # Handle environment commands
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
            return await environment.handle_command(command, context)
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error in {command.environment}: {str(e)}",
                suggested_commands=[f"{command.environment} help"],
                error=str(e)
            )
        
    def _add_to_history(self, role: str, content: str):
        """
        Add message to conversation history.
        Handles history trimming if needed.
        """
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        self.trim_conversation()

    def trim_conversation(self, max_length: int = Config.EXO_MAX_MESSAGES):
        """Trim conversation history while maintaining context."""
        if len(self.conversation_history) > max_length:
            # Keep system prompt and trim the rest
            self.conversation_history = [
                self.conversation_history[0],  # System prompt
                *self.conversation_history[-(max_length-1):]  # Recent messages
            ]

    async def _handle_pet_command(self, action: str) -> str:
        """
        Handle commands that interact with pet's internal systems.
        This could be split into a separate PetCommandHandler class if it grows complex.
        """
        # Example implementation - would need to be expanded based on available pet interactions
        parts = action.split(maxsplit=1)
        if not parts:
            return "Invalid pet command"

        action_type = parts[0]
        if action_type == "feed":
            # Interface with pet's needs system
            pass
        elif action_type == "play":
            # Interface with behavior system
            pass
        elif action_type == "status":
            # Get detailed internal state
            pass
        # ... other pet interactions

        return "Pet command handling not yet implemented"

    def trim_conversation(self, max_length: int = Config.EXO_MAX_MESSAGES):
        """Trim conversation history while maintaining context."""
        if len(self.conversation_history) > max_length:
            # Keep system prompt and trim the rest
            self.conversation_history = [
                self.conversation_history[0],  # System prompt
                *self.conversation_history[-(max_length-1):]  # Recent messages
            ]