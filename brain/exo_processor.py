"""
ExoProcessor: Manages the core exo-LLM loop logic, including command processing,
state integration, and maintaining conversation continuity.
"""

from typing import Dict, Any, List, Optional
from brain.command_preprocessor import CommandPreprocessor
from brain.environments.environment_registry import EnvironmentRegistry
from brain.environments.terminal_formatter import TerminalFormatter
from brain.logging_utils import ExoLogger
from config import Config
from datetime import datetime
import asyncio
import os

class ExoProcessor:
    def __init__(self, api_manager, state_bridge, environment_registry):
        self.api = api_manager
        self.state_bridge = state_bridge
        self.environment_registry = environment_registry
        self.command_preprocessor = CommandPreprocessor(self.api)
        self.conversation_history = []
        self._initialize_conversation()
        self.exo_lock = asyncio.Lock()
        self.last_successful_turn = None

    def _initialize_conversation(self):
        """Set up initial conversation context with welcome message."""
        initial_state = self.state_bridge.get_current_state()
        formatted_state = TerminalFormatter.format_terminal_view(initial_state)
        
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
        Returns command output if any action was taken.
        """
        ExoLogger.log_turn_start()

        if self.last_successful_turn and \
           (datetime.now() - self.last_successful_turn).total_seconds() < Config.EXO_MIN_INTERVAL:
            ExoLogger.log_turn_end(False, "Too soon since last completion")
            print("Skipping exo turn - too soon since last completion")
            return None

        try:
            async with asyncio.timeout(Config.EXO_TIMEOUT):
                if self.exo_lock.locked():
                    ExoLogger.log_turn_end(False, "Lock already held")
                    print("Exo turn already in progress, skipping")
                    return None
                    
                async with self.exo_lock:
                    try:
                        # Get LLM response
                        response = await self.api.openpipe.create_completion(
                            model=os.getenv("OPENPIPE_MODEL"),
                            messages=self.conversation_history,
                            temperature=Config.EXO_TEMPERATURE
                        )

                        llm_response = response["choices"][0]["message"]["content"]
                        ExoLogger.log_llm_exchange(self.conversation_history, llm_response)

                        #possibly swap this below and log the command instead into history; but in future, can have thinking tags maybe? unsure. would need advanced filtering.
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": llm_response
                        })

                        # Process command
                        command, help_text = await self.command_preprocessor.preprocess_command(
                            llm_response,
                            self.environment_registry.get_available_commands()
                        )

                        if not command:
                            return help_text
                        
                        ExoLogger.log_command_processing(response, command, help_text)

                        # Execute command and get response
                        command_response = await self._execute_command(command)
                        
                        # Format final response with state
                        current_state = await self.state_bridge.get_current_state()
                        formatted_response = TerminalFormatter.format_response(
                            command_response,
                            current_state
                        )

                        self.conversation_history.append({
                            "role": "user",
                            "content": formatted_response
                        })

                        self.last_successful_turn = datetime.now()
                        ExoLogger.log_turn_end(True)
                        return formatted_response
                    
                    except asyncio.TimeoutError:
                        ExoLogger.log_turn_end(False, "Exo loop run timeout")
                        print("Timeout in LLM request")
                        return None
                    except Exception as e:
                        ExoLogger.log_turn_end(False, str(e))
                        print(f"Error in exo turn: {e}")
                        return None
        except asyncio.TimeoutError:
            ExoLogger.log_turn_end(False, "Exo loop lock timeout")
            print("Timeout waiting for exo lock")
            return None
        except Exception as e:
            ExoLogger.log_turn_end(False, f"Fatal error: {str(e)}")
            print(f"Fatal error in exo turn: {e}")
            return None


    async def _execute_command(self, command: str) -> str:
        """Execute a command and return its response."""
        parts = command.split(maxsplit=1)
        if len(parts) < 2:
            if command.strip().lower() == "help":
                #need to implement this!!
                return self.environment_registry.format_global_help()
            return "Invalid command format. Use 'help' for available commands."

        env_name, action = parts

        # Handle internal pet commands
        if env_name == "pet":
            # This would interface with pet's internal modules for actions
            return await self._handle_pet_command(action)
            
        # Handle external environment commands
        environment = self.environment_registry.get_environment(env_name)
        if not environment:
            return f"Unknown environment: {env_name}"

        current_state = await self.state_bridge.get_current_state()
        return await environment.handle_command(action, current_state)

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